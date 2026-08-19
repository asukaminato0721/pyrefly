/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::check::Handles;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::report::scip;
use crate::state::require::Require;
use crate::state::state::State;

/// Generate a SCIP code navigation index.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct ScipArgs {
    /// Which files to index.
    #[command(flatten)]
    files: FilesArgs,

    /// Type checking arguments and configuration.
    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// Output SCIP protobuf file.
    #[arg(short = 'o', long, default_value = "index.scip")]
    output: PathBuf,
}

impl ScipArgs {
    pub fn run(
        self,
        version: &str,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files_to_check, config_finder, _) =
            self.files.resolve(self.config_override, wrapper)?;
        let expanded_file_list = config_finder.checkpoint(files_to_check.files_iter())?;
        let state = Forgetter::new(State::new(config_finder, thread_count), false);
        let handles = Handles::new(expanded_file_list);
        let (mut handles, _, sourcedb_errors) = handles.all(state.as_ref().config_finder());
        if !sourcedb_errors.is_empty() {
            for error in sourcedb_errors {
                error.print();
            }
            return Err(anyhow::anyhow!("Failed to query sourcedb."));
        }
        handles.sort_by_key(|handle| handle.path().dupe());

        let mut transaction = Forgetter::new(
            state.as_ref().new_transaction(Require::Everything, None),
            true,
        );
        transaction
            .as_mut()
            .run(&handles, Require::Everything, None);

        let project_root = std::env::current_dir()?.absolutize();
        let index = scip::index(transaction.as_ref(), &handles, &project_root, version)?;
        ::scip::write_message_to_file(&self.output, index)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(CommandExitStatus::Success)
    }
}
