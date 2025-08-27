#!/usr/bin/cmake -P

function(cget_exec)
    execute_process(${ARGN} RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL 0)
        message("${RESULT}")
        message(FATAL_ERROR "Process failed: ${ARGN}")
    endif()
endfunction()

set(ARGS)
foreach(i RANGE 3 ${CMAKE_ARGC})
    list(APPEND ARGS ${CMAKE_ARGV${i}})
endforeach()

include(CMakeParseArguments)

set(options help --minimum)
set(oneValueArgs --prefix)
set(multiValueArgs)

cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGS})

if(PARSE_help)
message("Usage: install_deps.cmake [options] [cmake-args]")
message("")
message("Options:")
message("  --prefix               Set the prefix to install the dependencies.")
message("")
message("Commands:")
message("  help                   Show this message and exit.")
message("")
message("  --minimum                  Install minimum dependencies.")
message("")
return()
endif()

set(_PREFIX /usr/local)
if(PARSE_--prefix)
    set(_PREFIX ${PARSE_--prefix})
endif()

get_filename_component(PREFIX ${_PREFIX} ABSOLUTE)

find_program(VIRTUALENV_PYTHON_EXE python3)
if(NOT VIRTUALENV_PYTHON_EXE)
    find_program(VIRTUALENV_PYTHON_EXE python)
endif()

function(virtualenv_create)
    cget_exec(
        COMMAND ${VIRTUALENV_PYTHON_EXE} -m venv ${PREFIX}
    )
endfunction()

function(virtualenv_install)
    virtualenv_create()
    # TODO: Check result
    message("${PREFIX}/pip install ${ARGN}")
    cget_exec(
        COMMAND ${PREFIX}/bin/python ${PREFIX}/bin/pip install ${ARGN}
    )
endfunction()

virtualenv_install(cget)

function(patch_cget_builder)
    # Get Python version for dynamic path
    execute_process(
        COMMAND ${PREFIX}/bin/python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
        COMMAND sed -i "s/multiprocessing.cpu_count()/8/g" "${PREFIX}/lib/${PYTHON_VERSION}/site-packages/cget/builder.py"
        RESULT_VARIABLE PATCH_RESULT
    )
    if(PATCH_RESULT EQUAL 0)
        message("Applied cget builder.py patch: limited multiprocessing.cpu_count() to 8 cores")
    else()
        message(WARNING "Failed to patch cget builder.py (may not exist yet)")
    endif()
endfunction()

# Apply cget builder.py patch
patch_cget_builder()


# Set compiler to hip-clang if not set
if(NOT DEFINED ENV{CXX} AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_program(CLANGXX clang++
        PATHS
            /opt/rocm/llvm
            /opt/rocm
        PATH_SUFFIXES
            bin
    )
    if(CLANGXX)
        set(ENV{CXX} ${CLANGXX})
    else()
        message(FATAL_ERROR "Cannot find clang++")
    endif()
endif()


if(NOT DEFINED ENV{CC} AND NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED CMAKE_TOOLCHAIN_FILE) 
    find_program(CLANGC clang
        PATHS
            /opt/rocm/llvm
            /opt/rocm
        PATH_SUFFIXES
            bin
    )
    if(CLANGC)
        set(ENV{CC} ${CLANGC})
    else()
        message(STATUS "Cannot find clang to set to CC")
    endif()
endif()

function(cget)
    cget_exec(COMMAND ${PREFIX}/bin/cget -p ${PREFIX} ${ARGN})
endfunction()

# Clean directory
# Skip clean since relative symlinks are incorrectly removed
# cget(clean -y)

set(TOOLCHAIN_FLAG)
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    set(TOOLCHAIN_FLAG -t ${CMAKE_TOOLCHAIN_FILE})
endif()
# Initialize directory
cget(init ${TOOLCHAIN_FLAG} -DCMAKE_INSTALL_RPATH=${PREFIX}/lib ${PARSE_UNPARSED_ARGUMENTS})
cget(ignore pcre)

# Install dependencies
cget(install -U ROCm/rocm-recipes@92c6695449c85887962f45509b376f2eb0d284f7)
cget(install -U -f requirements.txt)

# Install composable_kernel with Ninja generator
cget(install -U qiangpan2/composable_kernel@develop -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS="gfx1100" -DCK_TILE_USE_WMMA -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -G Ninja
)
