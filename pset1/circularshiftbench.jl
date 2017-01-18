##################################################
# problem 1 testing, benchmarking
# Corbin Foucart
##################################################

using Base.Test
using BenchmarkTools
using PyCall

# custom student submission type
type studentSubmission
    id::String
    testsPassed::Bool
    bmA::Array
    bmB::Array
    bmC::Array
end

# custom student submission constructor
studentSubmission(id) = studentSubmission(id, false, Float64[], Float64[], Float64[])

# 1D out of place circular shift testing function
function circularshift_test(X::AbstractVector, s::Integer)
    n = length(X)
    s = mod(s,n)
    return vcat(X[n-s+1:n], X[1:n-s])
end

# 1D in-place circular shift -- testing function
function circularshift_test!(X::AbstractVector, s::Integer, B=BitVector(length(X)))
    length(B) == length(X) || throw(DimensionMismatch("wrong number of bits"))
    fill!(B, false)
    n = length(X)
    for i = 1:n
        if !B[i] # cycle starting at i has not been visited yet
            Xi = X[i]
            j = i
            while true
                k = mod(j-1-s, n) + 1
                i == k && break # done with cycle
                X[j] = X[k]
                B[j] = true
                j = k
            end
            X[j] = Xi
            B[j] = true
        end
    end
    return X
end

# 2D circular shift -- test function
function circularshift_test!(X::AbstractMatrix, s::Integer)
    B = BitVector(size(X,2)) # allocate the bitvector only once for all rows
    for i = 1:size(X,1)
        circularshift_test!(view(X, i, :), s, B)
    end
    return X
end

# 2D circular shift -- test function
function circularshift_test!(X::AbstractMatrix)
    B = BitVector(size(X,2)) # allocate the bitvector only once for all rows
    for i = 1:size(X,1)
        circularshift_test!(view(X, i, :), i, B)
    end
    return X
end

# use the student's module to call a large suite of test cases, looking for errors
function correctness_tests(mod)

    # correctness test sizes
    testSizes = [2^n for n in 0:18];
    nShifts = 200
    testShifts = [collect(rand(-size:size, nShifts)) for size in testSizes];
    imgSizes = [2^n for n in 0:10];
    
    passing = true
    println("1D corner tests:")
    cN = 1000
    v = collect(1:cN)
    shifts = [0, cN, -1, -cN, 2*cN]
    testNms = ["zero shift", 
               "full length shift", 
               "negative shift", 
               "negative full shift", 
               "double shift"]
   
    try
        for test in collect(zip(shifts, testNms))
            (shift, tname) = test
            (e, sec_elapsed, bytes_alloc, secg) = @timed mod.circularshift!(v, shift)
            if (bytes_alloc > (128 + sizeof(v)/8.0))
                # do nothing, bogus measurement
            end
            @assert circularshift_test(copy(v), shift) == mod.circularshift!(v, shift)
            @printf "  %-25s | PASSED\n" tname
        end
    catch e
        @printf "  %-25s | FAILED\n" tname
        passing = false
    end
    
    @printf "1D fuzz tests:\n"
    overMemory = false
    try
        for (idx, size) in enumerate(testSizes)
            v = collect(1:size)
            shifts = vcat(testShifts[idx], testShifts[idx])
            for s in shifts
                @assert circularshift_test(copy(v), s) == mod.circularshift!(v, s)
            end
            (expr_val, sec_elapsed, bytes_alloc, sec_in_gc) = @timed mod.circularshift!(v, shifts[end])
            if (bytes_alloc > (128 + sizeof(v)/8.0))
                overMemory = true
                @assert (bytes_alloc < (128 + sizeof(v)/8.0))
            end
        end
        @printf "  %d %-20s | PASSED\n" length(testSizes)*nShifts "tests"
    catch e
        @printf "  %d %-20s | FAILED\n" length(testSizes)*nShifts "tests"
        if overMemory
            @printf "  * in-place memory limit exceeded!\n" 
        end
        passing = false
    end

    @printf "2D corner tests:\n" 
    tname = "rows >> cols"
    try 
        A = rand(1000, 10)
        @assert circularshift_test!(copy(A)) == mod.circularshift!(copy(A))
        @assert circularshift_test!(copy(A), 8) == mod.circularshift!(copy(A), 8)
        @printf "  %-25s | PASSED\n" tname
    catch e
        @printf "  %-25s | PASSED\n" tname
    end
    tname = "cols >> rows"
    try 
        A = rand(10, 1000)
        @assert circularshift_test!(copy(A)) == mod.circularshift!(copy(A))
        @assert circularshift_test!(copy(A), 8) == mod.circularshift!(copy(A), 8)
        @printf "  %-25s | PASSED\n" tname
    catch e
        @printf "  %-25s | FAILED\n" tname
        passing = false
    end

    @printf "2D fuzz tests\n" 
    try
        for (idx, size) in enumerate(imgSizes)
            A = rand(size, size)
            nShifts = 50
            shifts = rand(-size:size, nShifts)
            for shift in shifts
                @assert circularshift_test!(copy(A), shift) == mod.circularshift!(copy(A), shift) 
                @assert mod.circularshift!(copy(A)) == circularshift_test!(copy(A))
            end
        end
        @printf "  %d %-20s | PASSED\n" 2*length(imgSizes)*nShifts "tests"
    catch e
        @printf "  %d %-20s | FAILED\n" 2*length(imgSizes)*nShifts "tests"
        passing = false
    end

    tname = "AbstractArray test:"
    try
        # here we declare a numpy array object in julia
        # to test the code on an AbstractArray instance that is not an Array
        a = PyArray(pycall(pyimport("numpy.random")["rand"], PyObject, 3,4))
        mod.circularshift!(a, 2)
        @printf "%-27s | PASSED\n" tname
    catch e
        @printf "%-27s | FAILED\n" tname
        passing = false
    end

    tname = "empty array test (bonus):"
    try
        circularshift!(Float64[], 2)
        circularshift!(zeros(3, 0), 2)
        @printf "%-25s | PASSED\n" tname
    catch e
        @printf "%-27s | FAILED\n" tname
    end
    println("")

    return passing
end

# benchmark codes against a small group of test cases covering
# many potential use cases
function benchmark_tests(mod, s::studentSubmission)
    bmSizes1D = [10, 100, 1000, 10000, 1000000]
    bmSizes2D = [8,  64,  256,  500, 2000]
    dataTypes = [Float64, NTuple{16, Float64}, UInt8, Float64, Float64]
    smSz = 4

    for (ii, (size, size2D, dType)) in enumerate(collect(zip(bmSizes1D,
                                                             bmSizes2D,
                                                             dataTypes)))
        typeArray1D = Array{dType}(size)
        shifts = [5, round(Int64, size/4)]
        for sh in shifts
            elapsed = @elapsed mod.circularshift!(typeArray1D, sh)
            #trial = @benchmark mod.circularshift!(typeArray1D, sh)
            #elapsed = minimum(trial.times)
            push!(s.bmA, elapsed)

            sizeCombinations = [(size2D, smSz), (smSz, size2D), (size, size2D)]
            for dimComb in sizeCombinations
                typeArray2D = Array{dType}(dimComb)
                elapsed = @elapsed mod.circularshift!(typeArray2D, sh)
                #trial = @benchmark mod.circularshift!(typeArray2D, sh)
                #elapsed = minimum(trial.times)
                push!(s.bmB, elapsed)

                elapsed = @elapsed mod.circularshift!(typeArray2D)
                #trial = @benchmark mod.circularshift!(typeArray2D)
                #elapsed = minimum(trial.times)
                push!(s.bmC, elapsed)
            end
        end
    end
end

# computes the score for each participant by normalizing all times to the
# fastest time for each test, invert each, returning the median over the set of
# tests as the score.
function get_normed_scores(scores)

    # rows: student, cols: tests
    scoreTable = hcat(scores...)' 
    normalized = zeros(scoreTable)
    (nCompetitors, nTests) = size(scoreTable)
    for j = 1:nTests
        scoreCol = scoreTable[:,j]
        normalized[:, j] = 1./(scoreCol/minimum(scoreCol))
    end
    
    finalScores = Float64[]
    for i = 1:nCompetitors
        push!(finalScores, median(normalized[i, :]))
    end
    return finalScores
end

# loops over students to check, benchmark, and score the contest
function main()

    # iterate over student submissions and functions loaded to a module
    contestStudents = studentSubmission[]
    for fname in readdir("p1_code")
        if endswith(fname, ".jl")
            name = fname[1:end-3]
            mod = @eval module $(Symbol(name))
                  include($(joinpath("p1_code", fname)))
            end
        end

        # run correctness and benchmarks on the code
        println("------------------------------------")
        println("testing: $name")
        println("------------------------------------")
        submission = studentSubmission(name)
        submission.testsPassed = correctness_tests(mod)
        if submission.testsPassed
            println("All tests passed!")
            @printf "benchmarking..."
            benchmark_tests(mod, submission)
            @printf " done.\n\n"
            push!(contestStudents, submission)
        end
    end
    
    scoresA = get_normed_scores([student.bmA for student in contestStudents])
    scoresB = get_normed_scores([student.bmB for student in contestStudents])
    scoresC = get_normed_scores([student.bmC for student in contestStudents])

    println("contest results:\n")
    @printf "%-17s     a        b       c\n" "problem 1, part:"
    println("---------------------------------------------")
    for (idx, student) in enumerate(contestStudents)
        @printf "%-17s %4f %4f %4f \n" student.id scoresA[idx] scoresB[idx] scoresC[idx]
    end

end
main()

