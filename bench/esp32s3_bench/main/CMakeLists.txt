idf_component_register(SRCS "bench.c" "pie_kernels.S"
                       INCLUDE_DIRS "."
                       REQUIRES esp_partition esp_timer esp_hw_support esp_mm spi_flash)
