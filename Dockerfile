# Build local monorepo image
# docker build --no-cache -t  flowise .

# Run image
# docker run -d -p 3000:3000 flowise

FROM node:20-alpine

# Install system dependencies and build tools
RUN apk update && \
    apk add --no-cache \
        libc6-compat \
        python3 \
        make \
        g++ \
        build-base \
        cairo-dev \
        pango-dev \
        chromium \
        curl && \
    npm install -g pnpm

ENV PUPPETEER_SKIP_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

ENV NODE_OPTIONS=--max-old-space-size=12288

WORKDIR /usr/src/flowise

# Copy app source
COPY . .

# Install dependencies and build
RUN pnpm install && \
    pnpm build

# Give the node user ownership of the application files
RUN chown -R node:node .

# Switch to non-root user (node user already exists in node:20-alpine)
USER node

# Create serviceAccount.json from GSC_SERVICE_ACCOUNT build argument
ARG GSC_SERVICE_ACCOUNT
RUN echo "GSC_SERVICE_ACCOUNT length: ${#GSC_SERVICE_ACCOUNT}"
RUN echo "GSC_SERVICE_ACCOUNT first 100 chars: ${GSC_SERVICE_ACCOUNT:0:100}"
RUN mkdir -p packages/server/bin
RUN printf '%s\n' "$GSC_SERVICE_ACCOUNT" > packages/server/bin/serviceAccount.json


EXPOSE 3000

CMD [ "pnpm", "start" ]