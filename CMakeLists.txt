cmake_minimum_required(VERSION 3.16)

project(
  mcmc
  VERSION 2.1.0
  DESCRIPTION
    "Lightweight C++ library of Markov Chain Monte Carlo (MCMC) methods"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Build options
set(MCMC_LINEAR_ALGEBRA
    "EIGEN"
    CACHE STRING "Linear algebra backend to use (ARMA or EIGEN)")
set_property(CACHE MCMC_LINEAR_ALGEBRA PROPERTY STRINGS ARMA EIGEN)
option(MCMC_ENABLE_OPENMP "Enable OpenMP parallelisation" ON)
option(MCMC_HEADER_ONLY "Generate a header-only install tree" OFF)
option(EXAMPLES "Build example executables" OFF)

# Collect sources
set(MCMC_SOURCES
    src/aees.cpp
    src/de.cpp
    src/hmc.cpp
    src/mala.cpp
    src/nuts.cpp
    src/rmhmc.cpp
    src/rwmh.cpp)

if(MCMC_HEADER_ONLY)
  add_library(mcmc INTERFACE)
  set(MCMC_SCOPE INTERFACE)
else()
  add_library(mcmc ${MCMC_SOURCES})
  set(MCMC_SCOPE PUBLIC)
endif()

add_library(mcmc::mcmc ALIAS mcmc)

target_include_directories(
  mcmc ${MCMC_SCOPE} $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(mcmc ${MCMC_SCOPE} cxx_std_17)

if(MCMC_LINEAR_ALGEBRA STREQUAL "ARMA")
  if(NOT MCMC_HEADER_ONLY OR EXAMPLES)
    find_package(Armadillo REQUIRED)
  endif()
  if(NOT MCMC_HEADER_ONLY)
    target_link_libraries(mcmc ${MCMC_SCOPE} Armadillo::Armadillo)
  endif()
  target_compile_definitions(mcmc ${MCMC_SCOPE} MCMC_ENABLE_ARMA_WRAPPERS)
elseif(MCMC_LINEAR_ALGEBRA STREQUAL "EIGEN")
  if(NOT MCMC_HEADER_ONLY OR EXAMPLES)
    find_package(Eigen3 3.4 REQUIRED)
  endif()
  if(NOT MCMC_HEADER_ONLY)
    target_link_libraries(mcmc ${MCMC_SCOPE} Eigen3::Eigen)
  endif()
  target_compile_definitions(mcmc ${MCMC_SCOPE} MCMC_ENABLE_EIGEN_WRAPPERS)
else()
  message(
    FATAL_ERROR
      "Unsupported value '${MCMC_LINEAR_ALGEBRA}' for MCMC_LINEAR_ALGEBRA")
endif()

if(MCMC_ENABLE_OPENMP AND NOT MCMC_HEADER_ONLY)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(mcmc ${MCMC_SCOPE} OpenMP::OpenMP_CXX)
  else()
    message(
      WARNING
        "OpenMP requested but not found; continuing without OpenMP support")
  endif()
endif()

include(GNUInstallDirs)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  TARGETS mcmc
  EXPORT mcmcTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
  EXPORT mcmcTargets
  NAMESPACE mcmc::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mcmc)

include(CMakePackageConfigHelpers)

set(MCMC_PACKAGE_CONFIG_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mcmc)

configure_package_config_file(
  cmake/mcmcConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/mcmcConfig.cmake
  INSTALL_DESTINATION ${MCMC_PACKAGE_CONFIG_DESTINATION})

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/mcmcConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mcmcConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/mcmcConfigVersion.cmake
        DESTINATION ${MCMC_PACKAGE_CONFIG_DESTINATION})

export(
  EXPORT mcmcTargets
  NAMESPACE mcmc::
  FILE ${CMAKE_CURRENT_BINARY_DIR}/mcmcTargets.cmake)

if(EXAMPLES)
  add_subdirectory(examples)
endif()
