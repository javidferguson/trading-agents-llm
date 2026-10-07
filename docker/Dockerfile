# One image, three services. `decide`, `execute` and `dev` differ only by which
# dependency groups they install -- and by what they are allowed to reach.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

# THE VENV LIVES OUTSIDE /app, AND THAT IS LOAD-BEARING.
#
# The `dev` service bind-mounts the whole repo over /app so code edits take
# effect without a rebuild. With the default venv location (/app/.venv) that
# mount would shadow the interpreter and every installed package, and the
# container would die with "desk: not found" -- a confusing failure a long way
# from its cause. Putting it in /opt/venv makes the mount harmless.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# `execute` needs a real timezone database: IB stamps bars in the container's
# zone, and the ORB engine learned the hard way that a container defaulting to
# Etc/UTC produces bars an exchange session away from where you expect them.
#
# git and bash are for `dev`: an interactive shell that cannot run git is a
# shell you leave immediately.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl tzdata bash git less \
 && rm -rf /var/lib/apt/lists/*

# A login shell (`bash -l`, and some editors' integrated terminals) resets PATH
# from /etc/profile and drops the ENV above -- so `desk` and `pytest` vanish in
# the one workflow that is now the default. Verified: `bash -c` keeps
# /opt/venv/bin, `bash -lc` does not.
RUN printf 'export PATH="/opt/venv/bin:$PATH"\n' > /etc/profile.d/10-venv.sh

WORKDIR /app

# `execute` and `dev` install the ib_async extra; `decide` must NOT. Keeping
# ib_async out of decide's image is the coarsest and most reliable enforcement
# of the §0 split -- an accidental import fails at build time rather than at 3am.
ARG EXTRAS=""

# "--no-dev" for the production images; "" for `dev`, which needs pytest. The
# dev group also pulls ib_async so the carried safety tests can run.
ARG DEV_GROUP="--no-dev"

# LICENSE is here because pyproject.toml references it as the license file; the
# build fails without it, and the error ("License file does not exist") does not
# obviously point at a missing COPY.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
RUN uv sync --frozen ${DEV_GROUP} ${EXTRAS:+--extra $EXTRAS}

COPY config/ ./config/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

CMD ["desk", "doctor"]
