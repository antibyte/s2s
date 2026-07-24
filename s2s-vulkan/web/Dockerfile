# Lightweight static UI for testing s2s-vulkan (optional compose profile: web)
FROM nginx:1.27-alpine

# 00- so upstream loads before default server block. Re-resolve the Docker
# service name so a recreated backend does not leave nginx on a stale IP.
RUN printf 'upstream s2s_upstream {\n    zone s2s_upstream 64k;\n    resolver 127.0.0.11 valid=5s ipv6=off;\n    server s2s:8765 resolve;\n}\n' > /etc/nginx/conf.d/00-upstream.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css app.js /usr/share/nginx/html/

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
