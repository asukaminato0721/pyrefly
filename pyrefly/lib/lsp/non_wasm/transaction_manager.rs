/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_util::telemetry::TelemetryEvent;

use crate::state::require::Require;
use crate::state::state::CommittingTransaction;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::state::TransactionData;

/// `TransactionManager` aims to always produce a transaction that contains the up-to-date
/// in-memory contents.
#[derive(Default)]
pub struct TransactionManager<'a> {
    /// Invariant:
    /// If it's None, then the main `State` already contains up-to-date checked content
    /// of all in-memory files.
    /// Otherwise, it will contain up-to-date checked content of all in-memory files.
    saved_state: Option<TransactionData<'a>>,
}

impl<'a> TransactionManager<'a> {
    #[expect(clippy::result_large_err)] // Both results are basically the same size
    /// Produce a possibly committable transaction in order to recheck in-memory files.
    pub fn get_possibly_committable_transaction(
        &mut self,
        state: &'a State,
    ) -> Result<CommittingTransaction<'a>, Transaction<'a>> {
        // If there is no ongoing recheck due to on-disk changes, we should prefer to commit
        // the in-memory changes into the main state.
        if let Some(transaction) = state.try_new_committable_transaction(Require::Exports, None) {
            // If we can commit in-memory changes, then there is no point of holding the
            // non-committable transaction with a possibly outdated view of the `ReadableState`
            // so we can destroy the saved state.
            self.saved_state = None;
            Ok(transaction)
        } else {
            // If there is an ongoing recheck, trying to get a committable transaction will block
            // until the recheck is finished. This is bad for perceived perf. Therefore, we will
            // temporarily use a non-committable transaction to hold the information that's necessary
            // to power IDE services.
            Err(self.non_committable_transaction(state))
        }
    }

    /// Produce a `Transaction` to power readonly IDE services.
    /// This transaction will never be able to be committed.
    /// After using it, the state should be saved by calling the `save` method.
    ///
    /// The `Transaction` will always contain the handles of all open files with the latest content.
    /// It might be created fresh from state, or reused from previously saved state.
    ///
    /// If we were unable to restore a transaction from saved state, we create a fresh transaction.
    /// Callers may need to re-validate open files in this case.
    pub fn non_committable_transaction(&mut self, state: &'a State) -> Transaction<'a> {
        let previous_blocking = match self.saved_state.take() {
            Some(saved_state) => match saved_state.restore() {
                Ok(tx) => return tx,
                Err(blocked) => Some(blocked),
            },
            None => None,
        };
        let mut tx = state.transaction();
        tx.set_fresh();
        if let Some(d) = previous_blocking {
            tx.add_locked_blocking_duration(d);
        }
        tx
    }

    /// This function should be called once we finished using transaction for an LSP request.
    pub fn save(&mut self, transaction: Transaction<'a>, telemetry: &mut TelemetryEvent) {
        self.saved_state = Some(transaction.save(Some(telemetry)))
    }

    /// Save a transaction used outside the normal LSP telemetry lifecycle.
    pub fn save_without_telemetry(&mut self, transaction: Transaction<'a>) {
        self.saved_state = Some(transaction.save(None))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_build::handle::Handle;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_util::thread_pool::TEST_THREAD_COUNT;

    use super::TransactionManager;
    use crate::state::load::FileContents;
    use crate::state::require::Require;
    use crate::state::state::State;
    use crate::test::util::TestEnv;

    #[test]
    fn saved_answers_are_reused_only_for_the_current_state() {
        let mut env = TestEnv::new();
        env.add("query", "value = 1\n");
        env.add("main", "x = 1\n");
        let state = State::new(env.config_finder(), TEST_THREAD_COUNT);
        let handle = |name: &str| {
            Handle::new(
                ModuleName::from_str(name),
                ModulePath::memory(PathBuf::from(format!("{name}.py"))),
                env.sys_info(),
            )
        };
        let query = handle("query");
        let main = handle("main");
        let mut transaction = state.new_committable_transaction(Require::Exports, None);
        transaction.as_mut().set_memory(env.get_memory());
        state.run_with_committing_transaction(
            transaction,
            std::slice::from_ref(&main),
            Require::Everything,
            None,
            None,
        );
        let mut manager = TransactionManager::default();

        let mut transaction = manager.non_committable_transaction(&state);
        assert!(transaction.get_answers(&query).is_none());
        let runs = state.run_count();
        transaction.ensure_answers(&query, None);
        assert_eq!(state.run_count(), runs + 1);
        manager.save_without_telemetry(transaction);

        let mut transaction = manager.non_committable_transaction(&state);
        assert!(transaction.get_answers(&query).is_some());
        let runs = state.run_count();
        transaction.ensure_answers(&query, None);
        assert_eq!(state.run_count(), runs);
        manager.save_without_telemetry(transaction);

        let mut transaction = state.new_committable_transaction(Require::Exports, None);
        transaction.as_mut().set_memory(vec![(
            PathBuf::from("query.py"),
            Some(Arc::new(FileContents::from_source(
                "value = 'changed'".to_owned(),
            ))),
        )]);
        state.run_with_committing_transaction(
            transaction,
            &[main],
            Require::Everything,
            None,
            None,
        );

        let mut transaction = manager.non_committable_transaction(&state);
        assert!(transaction.get_answers(&query).is_none());
        let runs = state.run_count();
        transaction.ensure_answers(&query, None);
        assert_eq!(state.run_count(), runs + 1);
    }
}
