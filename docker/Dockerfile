# One image, two services. `decide` and `execute` differ by the extras they
# install and by what they are allowed to reach, not by their base layer.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# `execute` needs a real timezone database: IB stamps bars in the container's
# zone, and the ORB engine learned the hard way that a container defaulting to
# Etc/UTC produces bars an exchange session away from where you expect them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# `execute` installs the ib_async extra; `decide` must NOT. Keeping ib_async out
# of decide's image is the coarsest and most reliable enforcement of the §0
# split -- an accidental import fails at build time rather than at 3am.
ARG EXTRAS=""

# LICENSE is here because pyproject.toml references it as the license file; the
# build fails without it, and the error ("License file does not exist") does not
# obviously point at a missing COPY.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev ${EXTRAS:+--extra $EXTRAS}

COPY config/ ./config/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/

CMD ["desk", "doctor"]
