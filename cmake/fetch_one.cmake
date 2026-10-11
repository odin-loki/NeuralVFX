# Script mode (cmake -P): fetch one dependency with FetchContent into DIR/src. Run as a child process by
# cmake/encoders.cmake so that a failed download is an exit code there, not a fatal configure error.
#   -DNAME=name -DDIR=dir (-DGIT=url -DTAG=commit | -DURL=url -DSHA256=hash) [-DCMAKE_GENERATOR=... -DCMAKE_MAKE_PROGRAM=...]
include(FetchContent)
if(DEFINED URL)
  FetchContent_Populate(${NAME} QUIET SOURCE_DIR "${DIR}/src" BINARY_DIR "${DIR}/build" SUBBUILD_DIR "${DIR}/sub"
                        URL "${URL}" URL_HASH SHA256=${SHA256} DOWNLOAD_NO_EXTRACT TRUE TLS_VERIFY TRUE)
else()
  FetchContent_Populate(${NAME} QUIET SOURCE_DIR "${DIR}/src" BINARY_DIR "${DIR}/build" SUBBUILD_DIR "${DIR}/sub"
                        GIT_REPOSITORY "${GIT}" GIT_TAG "${TAG}" GIT_PROGRESS FALSE)
endif()
