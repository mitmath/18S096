##################################################
# Testing and benchmarking for pset 2, problem 1
# Corbin Foucart
##################################################

using BenchmarkTools

"""
Like `@benchmark`, but returns only the minimum time in ns.
"""
macro benchtime(args...)
    b = Expr(:macrocall, Symbol("@benchmark"), map(esc, args)...)
    :(time(minimum($b)))
end

# custom student submission type
type studentSubmission
    id::String
    bmP1::Array
    bmP2::Array
    bmP3::Array
end

# custom student submission constructor
studentSubmission(id::String) = studentSubmission(id, Float64[], Float64[], Float64[])

# turn our cf function into a macro
macro cf_check(x, a...)
    Expr(:call, :cf_check, x, a...)
end

# slow but correct sample implementation
function cf_check(x, a...)
    z = inv(one(x)) # initialize z this way for type stability
    for i = length(a):-1:1
        z = x + a[i] / z
    end
    return inv(z)
end

function mymatrix_check{T}(::Type{T}, m::Int)
    A = ones(T, m, m)
    for i = 1:m
        A[i,i] = 3
    end
    for i = 1:m-1
        A[i,i+1] = A[i+1,i] = 0
    end
    return A
end
      
mymatrix_check(m::Integer) = mymatrix_check(Int, Int(m))

# solve Ax = b, returning x, for the "mostly ones" matrix A above
function mysolve_check{T<:Number}(b::AbstractVector{T})
    m = length(b)
    A = mymatrix_check(float(T), m)
    return A \ b
end

# test for correctness via comparison to the slow but correct sample code.
# function returns a boolean detailing whether a student submission passes a
# test set.
function correctness_check(mod)

    passing  = [true, true, true]

    @printf "problem 1 tests:\n"
    @printf "\t%-25s" "type tests"
    try
       for dt in [Float64, UInt8, BigInt, Rational, Complex]
           ans = @cf_check one(dt) one(dt) one(dt)
           st_ans = @eval $(mod).@cf $(one(dt)) $(one(dt)) $(one(dt))
           @assert isapprox(ans, st_ans)
       end
       @printf "| PASSED\n"
    catch e
        @printf "| FAILED\n"
        passing[1] = false
    end

    @printf "\t%-25s" "variable length tests"
    try
      lengths = [4, 8, 10]
      for len in lengths
          a = rand(-100:100, len)
          cf_student(x) = @eval $(mod).@cf $(x) $(a...)
          cf_check(3.14, a...)
          @assert isapprox(cf_student(3.14), cf_check(3.14, a...))
      end
        @printf "| PASSED\n"
    catch e
        passing[1] = false
        @printf "| FAILED\n"
    end

    @printf "\t%-25s" "const test"
    try
        const ca1 = 5
        const ca2 = 7
        @eval $(mod).@cf $ca1 $ca2 $ca1
        @printf "| PASSED\n"
    catch e
        @printf "| FAILED\n"
        passing[1] = false
    end

    @printf "\t%-25s" "0, 1 args (bonus)"
    try
        @eval $(mod).@cf 1
        @eval $(mod).@cf 1 1
        @printf "| PASSED\n"
    catch e
        @printf "| FAILED\n"
    end

    @printf "problem 2 tests:\n"
    @printf "\t%-25s" "correctness checks: "
    try
        @assert broadcast(+, 1, 1:10) == @eval $(mod).mybroadcast(+, 1, 1:10)
        @assert broadcast(+, 1, 1:3, [10, 100, 1000]) == @eval $(mod).mybroadcast(+, 1, 1:3, [10, 100, 1000])
        @printf "| PASSED\n"
    catch e
        @printf "| FAILED\n"
        passing[2] = false
    end
    

    @printf "problem 3 tests:\n"
    @printf "\t%-25s" "correctness checks: "
    arg = zeros(Float64, 1000)
    try 
        @assert isapprox(mysolve_check(arg), @eval $(mod).mysolve($(arg)))
        @printf "| PASSED\n"
    catch e
        @printf "| FAILED\n"
        passing[3] = false
    end 

    return passing
end

function benchmark_submission!(s::studentSubmission, mod)
    @printf "benchmarking:\n"

    # problem 1
    @printf "\tP1..."
    lengths = [4, 8, 9, 10]
    testx = [3.14, 2 + 3im]
    for len in lengths
        for tx in testx
            a = rand(-10:10, len)
            cf_student = gensym()
            @eval $cf_student(x) = $(mod).@cf x $(a...)
            bm = @benchtime $cf_student($tx)
            push!(s.bmP1,bm)
        end 
    end
    const a1 = 5
    const a2 = 7
    a = [a1, a2]
    cf_student = gensym()
    @eval $cf_student(x) = $(mod).@cf x $(a...)
    bm = @benchtime $cf_student($a1)
    push!(s.bmP1,bm)
    @printf "done.\n"

    # problem 2
    @printf "\tP2..."
      push!(s.bmP2, @benchtime $(mod).mybroadcast(+, 1, 1:3, [10, 100, 1000]))
      push!(s.bmP2, @benchtime $(mod).mybroadcast(+, 1, 1:10, 3:12, 101:110, 1:10, 31:40, 61:70))
      push!(s.bmP2, @benchtime $(mod).mybroadcast(+, 1, $rand(20), $rand(20), $rand(20)))
      x = rand(1000)
      push!(s.bmP2, @benchtime $(mod).mybroadcast(+, $x, 1))

    @printf "done.\n"

    # problem 3
    @printf "\tP3..."
    dataTypes = [Float64, Float32, Int]
    for dt in dataTypes
        bm = @benchtime $(mod).mysolve($(zeros(dt, 1000)))
        push!(s.bmP3, bm)
    end
    @printf "done.\n"
    @printf "done\n\n"
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

# main body of testing
function main()

    contestStudents = studentSubmission[]
    for fname in readdir("student_code")
        if endswith(fname, ".jl")
            name = fname[1:end-3]
            moduleName = Symbol(name)
            mod = @eval module $(moduleName)
                include($(joinpath("student_code", fname)))
            end

            println("-----------------------------------------")
            println("testing: $name")
            println("-----------------------------------------")
            submission = studentSubmission(name)
            passed = correctness_check(moduleName)
            if all(passed)
                println("all tests passed!")
                benchmark_submission!(submission, moduleName)
                push!(contestStudents, submission)
            end
            println()
        end
    end

    scoresP1 = get_normed_scores([student.bmP1 for student in contestStudents])
    scoresP2 = get_normed_scores([student.bmP2 for student in contestStudents])
    scoresP3 = get_normed_scores([student.bmP3 for student in contestStudents])

    println("contest results:\n")
    @printf "%-17s     1        2       3\n" "problem:"
    println("---------------------------------------------")
    for (idx, student) in enumerate(contestStudents)
        @printf "%-17s %4f %4f %4f \n" student.id scoresP1[idx] scoresP2[idx] scoresP3[idx]
    end

end

main()
