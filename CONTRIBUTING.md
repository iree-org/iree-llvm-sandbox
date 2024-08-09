# How to Contribute

We'd love to accept your patches and contributions to this project.

To get started with contributing, please take a look at the
[Contributing](https://iree.dev/developers/general/contributing/) guide.

## Getting in touch

*   [GitHub issues](https://github.com/iree-org/iree-llvm-sandbox/issues): Feature requests,
    bugs, and other work tracking
*   [IREE Discord server](https://discord.gg/wEWh6Z9nMU): Daily development
    discussions with the core team and collaborators
*   [iree-discuss email list](https://groups.google.com/forum/#!forum/iree-discuss):
    Announcements, general and low-priority discussion

## Community guidelines

This project follows the
[LF Projects code of conduct](https://lfprojects.org/policies/code-of-conduct/).

## General purpose and spirit of this repository

The goal of this repository is to be a place for rapid prototyping and
cross-pollinating. This openness to experimentation and fast iteration should
not come at the cost of code quality: contributors are still expected to submit
minimal incremental patches that are easy to review and of comparable code
quality to MLIR core. When feasible, contributions should bias towards
end-to-end executable prototypes to fully demonstrate the value of a particular
idea and backing it by data. At the same time, compilers are complex beasts and
small incremental patches should be preferred; leniency is expected towards
"inflight" work that has been discussed. Contributors are strongly encouraged to
build with modularity and optionality in mind. Patches are expected to be single
purpose, self-contained and easy to revert. This is expected to reduce the churn
and simplify the implementation of a "decay process that" we discuss below.

The ideal end state for contributions to this repository is to either:

1.  become polished enough to be upstreamed to MLIR core and deleted from here.
2.  gradually decay and be deleted after a period of non-activity (e.g. 2-3
    months, responsibility of implementing the decay still TBD).
3.  find a new home as part of a bigger project if too specific and not
    generalizable enough (least preferred option).

For this purpose, we are considering splitting more experimental contributions
in subdirectories with the “root” being considered more fleshed out and likely
on track towards inclusion in MLIR core. As a strawman, let's say we have a
"root" whose structure mirrors MLIR core (i.e. `include`, `lib`, `test`,
`python` top-level dirs) complemented with an `experimental` directory (please
suggest a better name).

The constraints we are envisioning are open to suggestions and would resemble:

-   Root being on track to upstream needs to abide by MLIR coding standards and
    quality.
-   Formal RFCs are not needed for new concepts/dialects etc but discussions
    should still happen and a light form of consensus between a subset of
    contributors is expected.
-   Root is always green (CI story TBD).
-   Ideally the project would track LLVM at head and every contributor is
-   responsible of helping maintain root green (full details TBD).
-   `experimental/xxx` is owned by `xxx` and does not require consensus outside
    of `xxx` to land (e.g. `xxx==IREE`, `xxx=XLA`, `xxx=your_project_name`). An
    `experimental/xxx` directory is not even required to have a build defined.
-   Imports from the `xxx` directories to "root" are explicitly disallowed: the
    generalization and cleanup work needed to move into root undergoes
    comparable code reviews, testing and quality requirements.
-   Automatic build of `xxx` directories from the root and automatic git
    submodule that would trigger at root are explicitly disallowed: owners are
    expected to provide build and dependencies in their area and to keep their
    part green (or not, depending on constraints).
-   Ideally some sort of periodical evaluation of files that are bitroting and
    an automated warning that this seems to have reached end-of-life.
-   While the repository explicitly aims at biasing towards action, post-commit
    reviews are encouraged and commit authors are expected to address them.

In the spirit of velocity, we would like to propose a more dynamic and
interactive experience with people hopping on a call to discuss ideas, pair
program and pair review to fasttrack landing. If this mode were to be adopted,
such meetings would be recorded and a quick summary of the content and decisions
taken would be made available. This is a highly experimental idea to bias
towards action and prototyping, it is thus perfect for this repository :).
