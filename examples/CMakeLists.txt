cmake_minimum_required(VERSION 3.16)

if(MCMC_LINEAR_ALGEBRA STREQUAL "EIGEN")
    set(EIGEN_EXAMPLES
        aees_mixture
        de_normal_mean
        hmc_normal
        mala_normal
        nuts_normal
        rmhmc_normal
        rwmh_normal_mean
    )

    foreach(example ${EIGEN_EXAMPLES})
        set(target mcmc_example_eigen_${example})
        add_executable(${target} eigen/${example}.cpp)
        target_link_libraries(${target} PRIVATE mcmc)
        target_compile_features(${target} PRIVATE cxx_std_14)
        if(TARGET Eigen3::Eigen)
            target_link_libraries(${target} PRIVATE Eigen3::Eigen)
        endif()
    endforeach()
endif()

if(MCMC_LINEAR_ALGEBRA STREQUAL "ARMA")
    set(ARMADILLO_EXAMPLES
        aees_mixture
        de_normal_mean
        hmc_normal
        mala_normal
        nuts_normal
        rmhmc_normal
        rwmh_normal_mean
    )

    foreach(example ${ARMADILLO_EXAMPLES})
        set(target mcmc_example_armadillo_${example})
        add_executable(${target} armadillo/${example}.cpp)
        target_link_libraries(${target} PRIVATE mcmc)
        target_compile_features(${target} PRIVATE cxx_std_14)
        if(TARGET Armadillo::Armadillo)
            target_link_libraries(${target} PRIVATE Armadillo::Armadillo)
        endif()
    endforeach()
endif()
