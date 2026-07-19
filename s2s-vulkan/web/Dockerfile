# Lightweight static UI for testing s2s-vulkan (optional compose profile: web)
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css app.js /usr/share/nginx/html/

# Default upstream host for /ws proxy (overridden via env + entrypoint template if needed)
# Compose sets extra_hosts / network DNS name "s2s".
RUN printf 'upstream s2s_upstream {\n    server s2s:8765;\n}\n' > /etc/nginx/conf.d/upstream.conf

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
