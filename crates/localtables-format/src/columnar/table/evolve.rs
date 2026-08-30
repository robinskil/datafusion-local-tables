//! Schema changes: add, rename, drop, and retype a column.
//!
//! A change that leaves every stored byte meaning what it meant records a new
//! schema. A change that does not rewrites every segment in the same commit.

use super::*;

impl ColumnarTable {

    // ---- Schema changes -------------------------------------------------
    //
    // Two shapes.
    //
    // A change that leaves every stored byte meaning what it meant records a
    // new schema and nothing else.
    //
    // A change that does not rewrites every segment *in the same commit* as the
    // new schema. No instant exists where the schema says one thing and a
    // segment holds another.
    //
    // That keeps zone maps, filters and the zero-copy read path honest. A
    // segment always matches the schema in force.

    /// Add a column to the end of the schema.
    ///
    /// Rows already stored have no value for it and read as null, so the field
    /// must be nullable. Nothing is rewritten: this is one small commit
    /// whatever the table holds.
    ///
    /// The column goes at the end because anywhere else would move the columns
    /// after it, and a segment addresses its columns by position.
    pub async fn add_column(&self, field: FieldRef) -> Result<()> {
        if !field.is_nullable() {
            return Err(Error::InvalidArgument(format!(
                "cannot add a non-nullable column {}: the rows already stored \
                 have no value for it",
                field.name()
            )));
        }
        let current = self.schema();
        if current.index_of(field.name()).is_ok() {
            return Err(Error::InvalidArgument(format!(
                "the table already has a column named {}",
                field.name()
            )));
        }

        let mut fields: Vec<FieldRef> = current.fields().iter().cloned().collect();
        fields.push(field);
        self.set_schema(schema_with(&current, fields), false).await
    }


    /// Rename a column.
    ///
    /// Nothing is rewritten. A segment's bytes mean what its column *types*
    /// say, so a name is not part of what makes them readable.
    pub async fn rename_column(&self, from: &str, to: &str) -> Result<()> {
        let current = self.schema();
        let at = current.index_of(from).map_err(|_| {
            Error::InvalidArgument(format!("the table has no column named {from}"))
        })?;
        if from != to && current.index_of(to).is_ok() {
            return Err(Error::InvalidArgument(format!(
                "the table already has a column named {to}"
            )));
        }

        let mut fields: Vec<FieldRef> = current.fields().iter().cloned().collect();
        fields[at] = Arc::new(fields[at].as_ref().clone().with_name(to));
        self.set_schema(schema_with(&current, fields), false).await
    }


    /// Drop a column.
    ///
    /// Rewrites every segment, which is also what reclaims the column's bytes.
    /// A drop that only edited the schema would leave them on disk until
    /// something rewrote the table anyway.
    pub async fn drop_column(&self, name: &str) -> Result<()> {
        let current = self.schema();
        let at = current.index_of(name).map_err(|_| {
            Error::InvalidArgument(format!("the table has no column named {name}"))
        })?;
        if current.fields().len() == 1 {
            return Err(Error::InvalidArgument(
                "cannot drop the last column of a table".into(),
            ));
        }

        let mut fields: Vec<FieldRef> = current.fields().iter().cloned().collect();
        fields.remove(at);
        self.set_schema(schema_with(&current, fields), true).await
    }


    /// Change a column's type.
    ///
    /// This rewrites every segment and casts each one. Afterwards every segment
    /// holds the new type.
    ///
    /// The alternative is a cast at read time. That would cost the column its
    /// zero-copy path on every scan. It would also leave every zone map in the
    /// old type, and so unusable.
    ///
    /// A cast that cannot represent a stored value fails the whole change. It
    /// does not commit nulls in place of data.
    pub async fn cast_column(&self, name: &str, to: DataType) -> Result<()> {
        let current = self.schema();
        let at = current.index_of(name).map_err(|_| {
            Error::InvalidArgument(format!("the table has no column named {name}"))
        })?;
        if current.field(at).data_type() == &to {
            return Ok(());
        }
        if !arrow_cast::can_cast_types(current.field(at).data_type(), &to) {
            return Err(Error::InvalidArgument(format!(
                "cannot cast {name} from {} to {to}",
                current.field(at).data_type()
            )));
        }

        let mut fields: Vec<FieldRef> = current.fields().iter().cloned().collect();
        fields[at] = Arc::new(Field::new(name, to, fields[at].is_nullable()));
        self.set_schema(schema_with(&current, fields), true).await
    }


    /// Commit a new schema, rewriting the data first when it has to.
    ///
    /// `rewrite` says whether the stored bytes still mean what the new schema
    /// says.
    ///
    /// Where they do not, this reads every live row under the old schema,
    /// converts it, and writes it under the new one. The new segments and the
    /// new schema land in one commit.
    pub(super) async fn set_schema(&self, schema: SchemaRef, rewrite: bool) -> Result<()> {
        // Anything still in the memtable or the log is shaped by the old
        // schema. Landing it first means the change only has segments to think
        // about.
        self.flush().await?;

        let before = self.table_schema();
        let after = Arc::new(TableSchema::new(schema, &self.inner.options.cluster_by)?);

        let snapshot = self.snapshot();
        let sources: Vec<SegmentEntry> = snapshot.live_segments().cloned().collect();

        let mut writer = self.inner.writer.lock().await;
        if !writer.memtable.is_empty() {
            // A write landed between the flush above and this lock. Its rows
            // are shaped by the old schema, so the change is abandoned rather
            // than committed over them.
            return Err(Error::InvalidArgument(
                "a write landed while the schema was changing; try again".into(),
            ));
        }
        let mut manifest = writer.file.manifest().clone();
        manifest.txn_id = writer.file.meta().txn_id + 1;
        manifest.schema = writer.file.write_schema(&after.schema).await?;
        let min_active = self.inner.registry.min_active_txn();

        if rewrite {
            // Read under the old schema and write under the new one, a run of
            // segments at a time rather than the whole table at once. The
            // conversion has to reach disk in the same commit as the schema, so
            // this cannot be split across commits the way compaction is; what
            // it can do is bound what it holds while it works.
            //
            // The reads happen under the writer lock, unlike compaction's. A
            // schema change already refuses to run alongside a write, so there
            // is nothing to yield the lock for.
            let budget = self.inner.options.compaction_max_bytes.max(1);
            let rows: u64 = sources.iter().map(|entry| entry.row_count).sum();
            let group_rows = self.inner.options.row_group_size_for(rows);

            let written = async {
                let mut pending: Vec<RecordBatch> = Vec::new();
                let mut pending_bytes = 0u64;
                for entry in &sources {
                    for batch in self
                        .read_segment_as(&snapshot, entry, None, &before, None)
                        .await?
                    {
                        pending.push(convert(&batch, &before.schema, &after.schema)?);
                    }
                    pending_bytes += entry.data.len;
                    if pending_bytes < budget {
                        continue;
                    }
                    for group in
                        self.row_groups(std::mem::take(&mut pending), group_rows, &after)?
                    {
                        self.write_segment(&writer.file, &mut manifest, &group, &after, min_active)
                            .await?;
                    }
                    pending_bytes = 0;
                }
                for group in self.row_groups(pending, group_rows, &after)? {
                    self.write_segment(&writer.file, &mut manifest, &group, &after, min_active)
                        .await?;
                }
                Ok::<(), Error>(())
            }
            .await;
            written?;

            manifest
                .segments
                .retain(|entry| !sources.iter().any(|s| s.segment_id == entry.segment_id));
            for entry in &sources {
                manifest.free(entry.data);
                if let Some(dv) = entry.deletes {
                    manifest.free(dv);
                }
                writer.deletes.remove(&entry.segment_id);
                writer.dirty_deletes.remove(&entry.segment_id);
            }
        }

        match self.commit(&mut writer, manifest).await {
            Ok(()) => {
                // Three places hold the schema besides the manifest, and all of
                // them have to move together with it: the file, which is what a
                // snapshot copies; the memtable, which the next insert is
                // checked against and which a scan projects alongside the
                // segments; and the handle readers load from.
                writer.file.set_schema(after.schema.clone());
                writer.memtable =
                    Memtable::new(after.schema.clone(), writer.memtable.next_seqno());
                self.inner.schema.store(after);
                self.publish(&writer)?;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

/// The same schema with different fields, keeping its metadata.
pub(super) fn schema_with(schema: &SchemaRef, fields: Vec<FieldRef>) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// Reshape a batch read under `before` into one that fits `after`.
///
/// Columns are matched by name, so dropping one leaves the rest where they
/// belong rather than shifting them. A column the old schema does not have is
/// filled with nulls, which is the same thing an added column means.
pub(super) fn convert(batch: &RecordBatch, before: &SchemaRef, after: &SchemaRef) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(after.fields().len());
    for field in after.fields() {
        let column = match before.index_of(field.name()) {
            Ok(at) if batch.column(at).data_type() == field.data_type() => batch.column(at).clone(),
            Ok(at) => arrow_cast::cast(batch.column(at), field.data_type())?,
            Err(_) => arrow_array::new_null_array(field.data_type(), batch.num_rows()),
        };
        columns.push(column);
    }
    let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    RecordBatch::try_new_with_options(after.clone(), columns, &options).map_err(Error::from)
}
