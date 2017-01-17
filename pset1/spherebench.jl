##############################################################

# Solution testing/benchmarking code.   (Put student
# solutions in a directory "spherecode".   If a solution
# is too buggy to run, rename it to foo.jlbug.)

# Hack: for solutione whose makespheres just returns a Matrix{Float64},
# we wrap this in an immutable:
#   immutable Foo{T}
#       data::Matrix{T}
#   end
# and change the findsphere function to take a Foo argument.
# This way there is no ambiguity in dispatch of the findspheres
# function.  (A better solution would have been to not import
# Main.findsphere in the test code below, so that each module
# gets its own findsphere function.  Oh well, next time.)

##############################################################

# optimized linear search:
using StaticArrays
immutable Sphere3{T<:Real}
    center::SVector{3,T}
    radius2::T # the radius^2
end
Sphere3{T}(center::AbstractVector{T}, radius::T) = Sphere3{T}(SVector{3,T}(center), radius)
makespheres3{T<:Real}(data::AbstractMatrix{T}) = [Sphere3(data[i,1:3], data[i,4]^2) for i = 1:size(data,1)]
Base.in{T}(p::SVector{3,T}, S::Sphere3{T}) = (p[1]-S.center[1])^2 + (p[2]-S.center[2])^2 + (p[3]-S.center[3])^2 ≤ S.radius2
findsphere{T}(S::AbstractVector{Sphere3{T}}, p::SVector{3,T}) = findfirst(s -> p in s, S)
findsphere{T}(S::AbstractVector{Sphere3{T}}, p::AbstractVector) = findsphere(S, SVector{3,T}(p))

include("pset1-solutions.jl")

##############################################################

sols = Dict("KDTree" => makespheres,
            "Sphere3" => makespheres3)

for fname in readdir("spherecode")
    if endswith(fname, ".jl")
        name = fname[1:end-3]
        mod = @eval module $(Symbol(name))
            import Main: findsphere
            include($(joinpath("spherecode",fname)))
        end
        sols[name] = mod.makespheres
    end
end

##############################################################

testdata = randn(10000,4) .* [100 100 100 10]
testpoints = [randn(3) * 100 for i = 1:100];

spheres3 = makespheres3(testdata)
correct = map(p -> findsphere(spheres3, p), testpoints)
println("found: ", correct)

# spheres = Dict(n => ms(testdata) for (n,ms) in sols)
spheres = Dict()
for (n,ms) in sols
    spheres[n] = ms(testdata)
end
for name in keys(sols)
    print("Testing $name: ")
    if correct == map(p -> findsphere(spheres[name], p), testpoints)
        println("PASSED")
    else
        println("FAILED")
    end
    flush(STDOUT)
end

##############################################################

using BenchmarkTools

# search for a bunch of points ... return the total of the indices
function benchspheres(spheres, points)
    s = 0
    for i in eachindex(points)
        s += findsphere(spheres, points[i])
    end
    return s
end

times = Dict{String,Float64}()
for (name, S) in spheres
    b = @benchmark benchspheres($S, $testpoints)
    times[name] = time(minimum(b))
    println("$name time: ", times[name])
end
t0 = times["Sphere3"]
speeds = sort!(map(x -> (x[1],t0/x[2]), collect(times)), by=x->x[2])
for (n,s) in speeds
    println("$n speed: ", s)
end
