//! DataFusion table providers backed by a single local file.
//!
//! One file holds one table. Because the file is local, a scan can map it and
//! hand Arrow buffers to a query without copying them, which is the whole point
//! of the exercise.
//!
//! ```no_run
//! use datafusion::prelude::SessionContext;
//! use datafusion_local_tables::{register_columnar_table, TableOptions};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = SessionContext::new();
//! register_columnar_table(&ctx, "orders", "orders.lt".as_ref(), TableOptions::default())
//!     .await?;
//!
//! ctx.sql("SELECT * FROM orders WHERE id > 100").await?.show().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Everything a caller needs is re-exported here. A dependency on
//! `localtables-format` is not required.

pub mod columnar_exec;
pub mod columnar_provider;
pub mod dml;
pub mod pruning;

pub use columnar_exec::ColumnarScanExec;
pub use columnar_provider::{register_columnar_table, ColumnarTableProvider};
pub use dml::{ColumnarDataSink, DmlExec};
pub use localtables_format as format;
// Re-exported so a caller needs one crate and one `use` line. The options and
// the table itself both live in the format crate.
pub use localtables_format::{
    BloomFilters, ColumnarTable, Compression, Durability, IoBackend, TableOptions,
};
