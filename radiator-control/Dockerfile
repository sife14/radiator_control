# Home Assistant Add-on: Radiator Control MPC
# Multi-arch build für amd64, aarch64, armv7

ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python and dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-pandas \
    py3-scipy \
    build-base \
    python3-dev \
    && rm -rf /var/cache/apk/*

# Set working directory
WORKDIR /app

# Copy application - paths relative to add-on folder (build context)
COPY rootfs /
COPY src/ /app/src/
COPY main.py /app/
COPY requirements.txt /app/
COPY templates/ /app/templates/
COPY static/ /app/static/

# Install Python dependencies (--break-system-packages needed for PEP 668)
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Create data directory
RUN mkdir -p /data/radiator_control

# Labels
LABEL \
    io.hass.name="Radiator Control MPC" \
    io.hass.description="Intelligente Heizungssteuerung mit MPC" \
    io.hass.type="addon" \
    io.hass.version="1.1.0"

# Run
CMD ["/run.sh"]
