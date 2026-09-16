FROM nvcr.io/nvidia/isaac-sim@sha256:5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173
COPY wheels /opt/align-wheels
COPY runtime /opt/align-runtime
COPY OmniDrones /opt/OmniDrones
# Bootstrap uv from the hash-verified wheelhouse. Install only named locked additions;
# never resolve dependencies against, upgrade, or uninstall vendor torch/numpy/CUDA.
RUN /isaac-sim/python.sh -m pip install --no-index --no-deps --find-links=/opt/align-wheels uv==0.9.28 && \
    /isaac-sim/python.sh -m uv pip install --python /isaac-sim/kit/python/bin/python3 --no-index --find-links=/opt/align-wheels --no-deps --require-hashes -r /opt/align-runtime/additions.lock && \
    /isaac-sim/python.sh -m uv pip install --python /isaac-sim/kit/python/bin/python3 --no-deps --no-build-isolation --offline -e /opt/OmniDrones && \
    /isaac-sim/python.sh -m uv pip install --python /isaac-sim/kit/python/bin/python3 --no-deps --no-index --find-links=/opt/align-wheels align==0.1.0 && \
    /isaac-sim/python.sh /opt/align-runtime/verify_stack.py
ENV PYTHONUNBUFFERED=1 TZ=Asia/Kolkata
ENTRYPOINT ["/isaac-sim/python.sh"]
