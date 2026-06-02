/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Read;
use std::io::Write;
use std::net::Shutdown;
use std::net::TcpListener;
use std::net::TcpStream;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use pyrefly_util::thread_pool::ThreadCount;
use serde::Deserialize;
use serde::Serialize;
use tracing::info;
use tracing::warn;

use crate::commands::check::CheckResult;
use crate::commands::check::DaemonCheckResponse;
use crate::commands::check::DaemonCheckRunner;
use crate::commands::check::FullCheckArgs;
use crate::commands::util::CommandExitStatus;

pub const DEFAULT_DAEMON_ADDRESS: &str = "127.0.0.1:7820";

#[derive(Debug, Clone, Parser)]
pub struct DaemonArgs {
    /// Address to listen on for daemon check requests.
    #[arg(long, default_value = DEFAULT_DAEMON_ADDRESS)]
    address: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct DaemonRequest {
    cwd: PathBuf,
    args: FullCheckArgs,
}

#[derive(Debug, Deserialize, Serialize)]
enum DaemonResponse {
    Check(DaemonCheckResponse),
    Error(String),
}

pub fn run_check_client(
    args: FullCheckArgs,
) -> anyhow::Result<(CommandExitStatus, Option<CheckResult>)> {
    let request = DaemonRequest {
        cwd: std::env::current_dir().context("cannot identify current dir")?,
        args: args.clone(),
    };
    let mut stream = TcpStream::connect(args.daemon_address()).with_context(|| {
        format!(
            "could not connect to Pyrefly daemon at {}; start one with `pyrefly daemon --address {}`",
            args.daemon_address(),
            args.daemon_address()
        )
    })?;
    serde_json::to_writer(&mut stream, &request)?;
    stream.shutdown(Shutdown::Write)?;

    let response: DaemonResponse = serde_json::from_reader(stream)?;
    match response {
        DaemonResponse::Check(response) => {
            args.write_daemon_response(&response)?;
            Ok((response.status, Some(response.result)))
        }
        DaemonResponse::Error(error) => Err(anyhow::anyhow!(error)),
    }
}

impl DaemonArgs {
    pub async fn run(self, thread_count: ThreadCount) -> anyhow::Result<CommandExitStatus> {
        let listener = TcpListener::bind(&self.address)
            .with_context(|| format!("could not bind Pyrefly daemon to {}", self.address))?;
        info!("Pyrefly daemon listening on {}", self.address);
        let mut runner = DaemonCheckRunner::new(thread_count);
        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    if let Err(error) = handle_connection(&mut stream, &mut runner) {
                        warn!("Pyrefly daemon request failed: {error:#}");
                        let response = DaemonResponse::Error(format!("{error:#}"));
                        serde_json::to_writer(&mut stream, &response)?;
                        stream.flush()?;
                    }
                }
                Err(error) => warn!("Pyrefly daemon connection failed: {error}"),
            }
        }
        Ok(CommandExitStatus::Success)
    }
}

fn handle_connection(stream: &mut TcpStream, runner: &mut DaemonCheckRunner) -> anyhow::Result<()> {
    let mut body = String::new();
    stream.read_to_string(&mut body)?;
    let request: DaemonRequest = serde_json::from_str(&body)?;
    std::env::set_current_dir(&request.cwd)
        .with_context(|| format!("could not switch to request cwd {}", request.cwd.display()))?;
    let response = DaemonResponse::Check(runner.run(request.args)?);
    serde_json::to_writer(&mut *stream, &response)?;
    stream.flush()?;
    Ok(())
}
