FROM ubuntu:jammy

## Set locale
RUN apt-get update -qq \
    && apt-get install locales \
    && echo "en_GB.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen en_GB.utf8 \
    && /usr/sbin/update-locale LANG=en_GB.UTF-8
ENV LC_ALL en_GB.UTF-8
ENV LANG en_GB.UTF-8

## Set other env vars
ENV DEBIAN_FRONTEND noninteractive
ENV TZ UTC

## Setup R and r2u for fast package install
## Next RUN statement from r2u source, https://github.com/eddelbuettel/r2u/blob/master/inst/scripts/add_cranapt_jammy.sh
## Only change is to commands written to Rprofile.site (change order and add bspm.sudo = TRUE)
## First: update apt and get keys
RUN apt-get install --yes --no-install-recommends wget ca-certificates gnupg \
    && wget -q -O- https://eddelbuettel.github.io/r2u/assets/dirk_eddelbuettel_key.asc | tee -a /etc/apt/trusted.gpg.d/cranapt_key.asc \
## Second: add the repo -- here we use the well-connected mirror
    && echo "deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu jammy main" > /etc/apt/sources.list.d/cranapt.list \
## Third: ensure current R is used
    && wget -q -O- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && echo "deb [arch=amd64] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > /etc/apt/sources.list.d/cran_r.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 67C2D66C4B1D4339 51716619E084DAB9 \
    && apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends r-base-core \
## Fourth: add pinning to ensure package sorting
    && echo "Package: *" > /etc/apt/preferences.d/99cranapt \
    && echo "Pin: release o=CRAN-Apt Project" >> /etc/apt/preferences.d/99cranapt \
    && echo "Pin: release l=CRAN-Apt Packages" >> /etc/apt/preferences.d/99cranapt \
    && echo "Pin-Priority: 700"  >> /etc/apt/preferences.d/99cranapt \
## Fifth: install bspm (and its Python requirements) and enable it
## If needed (in bare container, say) install python tools for bspm and R itself
    && apt-get install --yes --no-install-recommends python3-dbus python3-gi python3-apt make \
## Then install bspm (as root) and enable it, and enable a speed optimization
    && Rscript -e 'install.packages("bspm")' \
    && RHOME=$(R RHOME) \
    && echo "options(bspm.version.check = FALSE, bspm.sudo = TRUE)" >> ${RHOME}/etc/Rprofile.site \
    && echo "suppressMessages(bspm::enable())" >> ${RHOME}/etc/Rprofile.site

## Install git and RStudio Server
RUN apt-get install --yes --no-install-recommends gdebi-core git-all \
    && wget -q https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2023.06.1-524-amd64.deb \
    && apt-get install --yes --no-install-recommends ./rstudio-server-2023.06.1-524-amd64.deb \
    && rm rstudio-server-2023.06.1-524-amd64.deb

## Next, setup user and sudo privileges
RUN apt-get install --yes --no-install-recommends sudo \
    && useradd -m -s /bin/bash rstudio \
    && echo "rstudio ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
# Disable login requirements for RStudio and ensure only listens on localhost
    && echo "auth-none=1\nwww-address=127.0.0.1\nwww-port=8787" >> /etc/rstudio/rserver.conf \
    && echo "USER=rstudio" >> /etc/environment \
# Enable rstudio to write to packages folder
    && chown root:rstudio "/usr/lib/R/site-library" \
    && chmod g+ws "/usr/lib/R/site-library"

## APT config & tidyup
RUN echo 'APT::Install-Recommends "false";' > /etc/apt/apt.conf.d/90local-no-recommends \
    && rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && rm -rf /var/lib/apt/lists/*
