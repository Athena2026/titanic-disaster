# Use R base image
FROM r-base:4.3.1

# Copy install_packages.R and run it
COPY install_packages.R /r_app/install_packages.R
RUN Rscript /r_app/install_packages.R

# Copy main.R script
COPY main.R /r_app/main.R

# Set working directory
WORKDIR /r_app

# Command to run
CMD ["Rscript", "main.R"]