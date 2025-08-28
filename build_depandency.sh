#!/bin/bash

mkdir -p /tmp/fake_nproc
echo '#!/bin/bash' > /tmp/fake_nproc/nproc
echo 'echo 8' >> /tmp/fake_nproc/nproc
chmod +x /tmp/fake_nproc/nproc

clean_build_cache() {
    echo "Cleaning build cache only..."
    
    local DEPS_PREFIX="${HOME}/miopen-deps"
    
    rm -rf "${DEPS_PREFIX}"
    rm -rf build
    echo "Build cache cleaned."
}
clean_build_cache

clean_environment() {

    export PATH=$(echo "$PATH" | tr ':;' '\n' | grep '^/' | grep -v '\\' | grep -v '^[A-Za-z]:')
    export PATH=$(echo "$PATH" | tr '\n' ':' | sed 's/:$//')
    export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

    unset VSCODE_IPC_HOOK_CLI
    unset VSCODE_GIT_ASKPASS_NODE
    unset VSCODE_GIT_ASKPASS_EXTRA_ARGS
    unset VSCODE_GIT_ASKPASS_MAIN
    unset VSCODE_GIT_IPC_HANDLE
    unset VSCODE_INJECTION

    unset APPDATA
    unset LOCALAPPDATA
    unset USERPROFILE
    unset PROGRAMFILES
    unset PROGRAMDATA
    unset TEMP
    unset TMP

    echo "Environment cleaned. Current PATH:"
    echo "$PATH" | tr ':' '\n' | head -10
}

clean_environment

PATH="/tmp/fake_nproc:$PATH"

DEPS_PREFIX="${HOME}/miopen-deps"

echo "Creating build directories..."
mkdir -p build
mkdir -p "${DEPS_PREFIX}"

echo "Installing dependencies to ${DEPS_PREFIX}..."
cmake -P install_deps.cmake --prefix "${DEPS_PREFIX}"

#for CK build update standalone
#${HOME}/miopen-deps/bin/cget -p ${HOME}/miopen-deps install -U qiangpan2/composable_kernel@develop -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS="gfx1100" -DCK_TILE_USE_WMMA=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -G Ninja
