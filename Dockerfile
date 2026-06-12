# llmtrim proxy image — static musl binary on a distroless base (CA certs included for
# upstream TLS; no shell, no package manager). Built in release.yml from the released
# binary, not from source: the image ships exactly the attested artifact.
#
#   docker run -d -p 43117:43117 -v llmtrim-state:/data ghcr.io/fkiene/llmtrim
#
# TARGETARCH is set by buildx (amd64 → x86_64-musl asset, arm64 → aarch64-gnu asset);
# the per-arch binary is staged by the workflow into binaries/<arch>/llmtrim.
FROM gcr.io/distroless/cc-debian12:nonroot
ARG TARGETARCH
COPY binaries/${TARGETARCH}/llmtrim /usr/local/bin/llmtrim
ENV HOME=/data XDG_DATA_HOME=/data XDG_CONFIG_HOME=/data LLMTRIM_BIND=0.0.0.0
VOLUME /data
EXPOSE 43117
ENTRYPOINT ["/usr/local/bin/llmtrim"]
CMD ["serve"]
