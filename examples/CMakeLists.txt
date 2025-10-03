cmake_minimum_required(VERSION 3.16)

set(_examples_subdir "")
set(_examples_dependency "")

if(MCMC_LINEAR_ALGEBRA STREQUAL "EIGEN")
  set(_examples_subdir eigen)
  if(TARGET Eigen3::Eigen)
    set(_examples_dependency Eigen3::Eigen)
  endif()
elseif(MCMC_LINEAR_ALGEBRA STREQUAL "ARMA")
  set(_examples_subdir armadillo)
  if(TARGET Armadillo::Armadillo)
    set(_examples_dependency Armadillo::Armadillo)
  endif()
else()
  message(WARNING "Unsupported backend '${MCMC_LINEAR_ALGEBRA}' for examples")
  return()
endif()

file(GLOB _examples_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/${_examples_subdir}/*.cpp")

if(NOT _examples_sources)
  message(WARNING "No example sources found in '${_examples_subdir}'")
  return()
endif()

foreach(_example_source IN LISTS _examples_sources)
  get_filename_component(_example_name "${_example_source}" NAME_WE)

  add_executable(${_example_name} "${_example_source}")
  target_link_libraries(${_example_name} PRIVATE mcmc)
  target_compile_features(${_example_name} PRIVATE cxx_std_14)

  if(_examples_dependency)
    target_link_libraries(${_example_name} PRIVATE ${_examples_dependency})
  endif()
endforeach()

unset(_examples_subdir)
unset(_examples_dependency)
unset(_examples_sources)
