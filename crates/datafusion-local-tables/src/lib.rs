//! DataFusion table providers backed by a single local file.
//!
//! One file holds one table. Because the file is local, a scan can map it and
//! hand Arrow buffers to a query without copying them, which is the whole point
//! of the exercise.
//!
//! ```no_run
//! use datafusion::prelude::SessionContext;
//! use datafusion_local_tables::{ColumnarTableProvider, format::TableOptions};
//! use localtables_format::ColumnarTable;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let table = ColumnarTable::open("orders.lt".as_ref(), TableOptions::default()).await?;
//! let ctx = SessionContext::new();
//! ctx.register_table("orders", Arc::new(ColumnarTableProvider::new(table)))?;
//!
//! let df = ctx.sql("SELECT * FROM orders WHERE id > 100").await?;
//! df.show().await?;
//! # Ok(())
//! # }
//! ```

pub mod btree_exec;
pub mod btree_provider;
pub mod columnar_exec;
pub mod columnar_provider;
pub mod dml;
pub mod pruning;

pub use btree_exec::{BTreeScanExec, KeyRange};
pub use btree_provider::{register_btree_table, BTreeTableProvider};
pub use columnar_exec::ColumnarScanExec;
pub use columnar_provider::{register_columnar_table, ColumnarTableProvider};
pub use dml::{ColumnarDataSink, DmlExec};
pub use localtables_format as format;
pub use localtables_format::btree::BTreeTable;
pub use localtables_format::ColumnarTable;
