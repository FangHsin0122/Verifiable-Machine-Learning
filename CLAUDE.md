# EECE 7268 VML repo

## Git branches

**Never merge `private` into `main`, and never push `private`.**

- `main` is the public branch. It's pushed to GitHub (`origin`), which is a public repo.
- `private` is local only. It tracks files we don't publish: classmates' work, class rosters, AI-assisted drafts, `README.local.md` files, and job-search notes.
- Merging `main` into `private` is fine. That's how `private` stays up to date.
- Merging `private` into `main`, or pushing `private` (`git push origin private`, `git push --all`), would put those files on GitHub. If that happens, removing them takes a history rewrite and a force push.
- We commit public work on `main` (`git checkout main` first) and keep local-only files on `private`.
- Before pushing `main`, we check that `git ls-tree -r --name-only main` lists no classmates' files, AI drafts, or `README.local.md`.
