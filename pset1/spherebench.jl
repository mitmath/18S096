##############################################################

# Solution testing/benchmarking code.   (Put student
# solutions in a directory "spherecode".)

##############################################################

# optimized linear search:
module S3
using StaticArrays
immutable Sphere3{T<:Real}
    center::SVector{3,T}
    radius2::T # the radius^2
end
Sphere3{T}(center::AbstractVector{T}, radius::T) = Sphere3{T}(SVector{3,T}(center), radius)
makespheres{T<:Real}(data::AbstractMatrix{T}) = [Sphere3(data[i,1:3], data[i,4]^2) for i = 1:size(data,1)]
Base.in{T}(p::SVector{3,T}, S::Sphere3{T}) = (p[1]-S.center[1])^2 + (p[2]-S.center[2])^2 + (p[3]-S.center[3])^2 ≤ S.radius2
findsphere{T}(S::AbstractVector{Sphere3{T}}, p::SVector{3,T}) = findfirst(s -> p in s, S)
findsphere{T}(S::AbstractVector{Sphere3{T}}, p::AbstractVector) = findsphere(S, SVector{3,T}(p))
end

module KD
  using StaticArrays
  include("pset1-solutions.jl")
end

##############################################################

sols = Dict("KDTree" => KD,
            "Sphere3" => S3)

for fname in readdir("spherecode")
    if endswith(fname, ".jl")
        name = fname[1:end-3]
        mod = @eval module $(Symbol(name))
            include($(joinpath("spherecode",fname)))
        end
        sols[name] = mod
    end
end

##############################################################

testdata = randn(10000,4) .* [100 100 100 10]
# don't allow negative radii in test data:
testdata[:,4] = abs(testdata[:,4])
testpoints = [randn(3) * 100 for i = 1:100];

spheres3 = S3.makespheres(testdata)
correct = map(p -> S3.findsphere(spheres3, p), testpoints)
println("found: ", correct)

# spheres = Dict(n => ms(testdata) for (n,ms) in sols)
spheres = Dict()
for (n,mod) in sols
    try
        spheres[n] = mod.makespheres(testdata)
    catch e
        println("$n: ERROR $e in makespheres")
    end
end
for name in keys(spheres)
    print("Testing $name: ")
    try
        mod = sols[name]
        got = map(p -> mod.findsphere(spheres[name], p), testpoints)
        if correct == got
            println("PASSED")
        else
            println("FAILED ", countnz(correct .!= got), "/", length(correct), " tests")
            println("got: ", got)
            println("passed: ", correct .== got)
        end
    catch e
        println("ERROR $e")
    end
    flush(STDOUT)
end

##############################################################

using BenchmarkTools

# search for a bunch of points ... return the total of the indices
function benchspheres(findsphere, spheres, points)
    s = 0
    for i in eachindex(points)
        s += findsphere(spheres, points[i])
    end
    return s
end

times = Dict{String,Float64}()
for (name, S) in spheres
    try
        f = sols[name].findsphere
        b = @benchmark benchspheres($f, $S, $testpoints)
        times[name] = time(minimum(b))
        println("$name time: ", times[name])
    catch
        # ignore exceptions
    end
end
t0 = times["Sphere3"]
speeds = sort!(map(x -> (x[1],t0/x[2]), collect(times)), by=x->x[2])
for (n,s) in speeds
    println("$n speed: ", s)
end
