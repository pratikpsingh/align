# ALiGn documentation reading order

Start here, then follow the numbered files in order. The prefixes describe reading order, not implementation stages.

1. [Installation](01-installation.md): prepare the uv environment and understand laptop/lab requirements.
2. [Diagnostics](02-diagnostics.md): collect machine information and interpret the report.
3. [GPU container setup](03-gpu-container-setup.md): prepare Docker and NVIDIA access on the lab machine.
4. [Isaac Sim startup check](04-isaac-sim-smoke.md): test the candidate simulator and save evidence.
5. [Development](05-development.md): understand the package structure, coding workflow, and local checks.

6. [Lab handoff](06-lab-handoff.md): verified simulator result and a detailed agent continuation prompt.

7. [OmniDrones runtime](07-omnidrones-runtime.md): exact source/dependency pins, Python boundary, image build and backup.
8. [Single-drone control](08-single-drone-control.md): frames, actions, controller/reset contract, measurements and lab evidence.

For beginner explanations, begin with the [learning index](../learning/00-README.md). For research scope and intended implementation, use the [planning index](../plans/00-README.md). Setup instructions distinguish implemented tools from pending lab validation.
