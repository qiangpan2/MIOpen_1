#!/bin/bash

# 设置默认的安装前缀
mkdir -p /tmp/fake_nproc
echo '#!/bin/bash' > /tmp/fake_nproc/nproc
echo 'echo 8' >> /tmp/fake_nproc/nproc
chmod +x /tmp/fake_nproc/nproc

clean_build_cache() {
    echo "Cleaning build cache only..."
    
    local DEPS_PREFIX="${HOME}/miopen-deps"
    
    # 只清理构建缓存，保留已安装的包
    rm -rf "${DEPS_PREFIX}"
    rm -rf build
    echo "Build cache cleaned."
}
clean_build_cache

clean_environment() {
    # 清理PATH中的Windows路径（包含驱动器字母和反斜杠的路径）
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^[A-Za-z]:' | grep -v '\\' | tr '\n' ':' | sed 's/:$//')
    
    # 确保基本Linux路径存在
    export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
    
    # 清理VS Code相关的环境变量（这些经常包含Windows路径）
    unset VSCODE_IPC_HOOK_CLI
    unset VSCODE_GIT_ASKPASS_NODE
    unset VSCODE_GIT_ASKPASS_EXTRA_ARGS
    unset VSCODE_GIT_ASKPASS_MAIN
    unset VSCODE_GIT_IPC_HANDLE
    unset VSCODE_INJECTION
    
    # 清理其他可能的Windows环境变量
    unset APPDATA
    unset LOCALAPPDATA
    unset USERPROFILE
    unset PROGRAMFILES
    unset PROGRAMDATA
    unset TEMP
    unset TMP
    
    # 确保必要的构建工具在PATH中
    if [[ -d "/opt/rocm/llvm/bin" ]]; then
        export PATH="/opt/rocm/llvm/bin:$PATH"
    fi
    
    if [[ -d "/opt/rocm/bin" ]]; then
        export PATH="/opt/rocm/bin:$PATH"
    fi
    
    echo "Environment cleaned. Current PATH:"
    echo "$PATH" | tr ':' '\n' | head -10
}

# 调用清理函数
clean_environment

PATH="/tmp/fake_nproc:$PATH"

DEPS_PREFIX="${HOME}/miopen-deps"

echo "Creating build directories..."
mkdir -p build
mkdir -p "${DEPS_PREFIX}"

echo "Installing dependencies to ${DEPS_PREFIX}..."
cmake -P install_deps.cmake --prefix "${DEPS_PREFIX}"

