FROM ghcr.io/rocker-org/devcontainer/tidyverse:4.3

# Disable login requirements for RStudio
RUN echo "auth-none=1" >> /etc/rstudio/rserver.conf \
    && echo "USER=rstudio" >> /etc/environment
