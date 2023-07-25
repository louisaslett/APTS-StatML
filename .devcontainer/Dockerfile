FROM ghcr.io/rocker-org/devcontainer/tidyverse:4.3

# Disable login requirements for RStudio
RUN echo "auth-none=1" >> /etc/rstudio/rserver.conf \
    && echo "USER=rstudio" >> /etc/environment

## Set locale
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
	&& locale-gen en_GB.utf8 \
	&& /usr/sbin/update-locale LANG=en_GB.UTF-8

ENV LC_ALL en_GB.UTF-8
ENV LANG en_GB.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV TZ UTC

# Install r2u
RUN apt-get update -qq \
    && apt-get install --yes --no-install-recommends wget ca-certificates gnupg \
    && wget -q -O- https://eddelbuettel.github.io/r2u/assets/dirk_eddelbuettel_key.asc | tee -a /etc/apt/trusted.gpg.d/cranapt_key.asc \
    && echo "deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu jammy main" > /etc/apt/sources.list.d/cranapt.list \
    && echo "Package: *" > /etc/apt/preferences.d/99cranapt \
    && echo "Pin: release o=CRAN-Apt Project" >> /etc/apt/preferences.d/99cranapt \
    && echo "Pin: release l=CRAN-Apt Packages" >> /etc/apt/preferences.d/99cranapt \
    && echo "Pin-Priority: 700"  >> /etc/apt/preferences.d/99cranapt \
    && apt-get update -qq \
    && apt-get install --yes --no-install-recommends r-cran-bspm r-cran-docopt r-cran-littler r-cran-remotes python3-dbus python3-gi python3-apt \
    && ln -s /usr/lib/R/site-library/littler/examples/install.r /usr/local/bin/install.r \
    && ln -s /usr/lib/R/site-library/littler/examples/install2.r /usr/local/bin/install2.r \
    && ln -s /usr/lib/R/site-library/littler/examples/installBioc.r /usr/local/bin/installBioc.r \
    && ln -s /usr/lib/R/site-library/littler/examples/installDeps.r /usr/local/bin/installDeps.r \
    && ln -s /usr/lib/R/site-library/littler/examples/installGithub.r /usr/local/bin/installGithub.r \
    && ln -s /usr/lib/R/site-library/littler/examples/installRub.r /usr/local/bin/installRub.r \
    && ln -s /usr/lib/R/site-library/littler/examples/testInstalled.r /usr/local/bin/testInstalled.r \
    && ln -s /usr/lib/R/site-library/littler/examples/update.r /usr/local/bin/update.r \
    && echo "suppressMessages(bspm::enable())" >> /etc/R/Rprofile.site \
    && echo "options(bspm.version.check=FALSE)" >> /etc/R/Rprofile.site \
    && echo 'APT::Install-Recommends "false";' > /etc/apt/apt.conf.d/90local-no-recommends \
    && rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && rm -rf /var/lib/apt/lists/*
