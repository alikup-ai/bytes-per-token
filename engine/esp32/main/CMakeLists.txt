idf_component_register(SRCS "main.c" "demo.c" "pie_llm.S" "../../src/llm.c"
                       INCLUDE_DIRS "." "../../src"
                       REQUIRES esp_partition esp_hw_support spi_flash esp_driver_uart)
# Bit-identical floats with the PC build: no fused multiply-add (madd.s).
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
# Live demo firmware: idf.py -B builddemo -DLLM_DEMO=1 build (engine/demo/README.md)
if(LLM_DEMO)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE LLM_DEMO=1)
endif()
