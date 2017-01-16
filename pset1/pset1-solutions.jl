# This file contains just the final implementations from the solutions
# notebook, with no tests, explanations, or benchmarks.   Refer to the
# pset1-solutions.ipynb notebook for more information.

########################################################################
# problem 1(a)

function circshift_perm2!(X::AbstractVector, s::Int, B=BitVector(s))
    length(B) == s || throw(DimensionMismatch("wrong number of bits"))
    fill!(B, false)
    n = length(X)
    @assert 0 ≤ s < n
    s == 0 && return X
    ncycles = gcd(s,n)
    @inbounds for i = 1:s
        if !B[i] # cycle starting at i has not been visited yet
            Xi = X[i]
            j = i
            while true
                k = j - s + n
                X[j] = X[k]
                B[j] = true
                j = k
                while j > s
                    k = j - s # j should be > s
                    X[j] = X[k]
                    j = k
                end
                j == i && break # end of cycle
            end
            k = j + s
            X[k > n ? k - n : k] = Xi
            ncycles -= 1
            ncycles == 0 && break
        end
    end
    return X
end

function circshift_reverse!(a::AbstractVector, s::Int)
    n = length(a)
    # these optimizations are implemented below
    # s = mod(s, n)
    # s == 0 && return a
    @assert 0 ≤ s < n
    s′ = n-s
    reverse!(a, 1, s′)
    reverse!(a, s′+1, n)
    reverse!(a)
end

# I would use the copy! function here, which is highly optimized, but I
# ran into a problem with views: https://github.com/JuliaLang/julia/issues/20069
function mycopy!(a, s, s′)
    @simd for i = s′:-1:1 # copy a[1:s′] to a[s+1:n]
        @inbounds a[s+i] = a[i]
    end
end
mycopy!(a::Array, s, s′) = copy!(a, s+1, a, 1, s′)

function circshift_buf1!{T}(a::AbstractVector{T}, buf::AbstractVector{T})
    n = length(a)
    s = length(buf)
    s′ = n - s
    copy!(buf, 1, a, s′+1, s) # copy a[s′+1:n] to buf[1:s]
    mycopy!(a, s, s′) # copy a[1:s′] to a[s+1:n]
    copy!(a, 1, buf, 1, s)    # copy buf to a[1:s]
    return a
end
circshift_buf1!{T}(a::AbstractVector{T}, s::Int) = circshift_buf1!(a, Array{T}(s))

function circshift_buf2!{T}(a::AbstractVector{T}, buf::AbstractVector{T})
    n = length(a)
    s′ = length(buf)
    s = n - s′
    copy!(buf, 1, a, 1, s′)   # copy a[1:s′] to buf
    copy!(a, 1, a, s′+1, s)   # copy a[s′+1:n] to a[1:s]
    copy!(a, s+1, buf, 1, s′) # copy buf to a[s+1:n]
    return a
end
circshift_buf2!{T}(a::AbstractVector{T}, s::Int) = circshift_buf2!(a, Array{T}(length(a)-s))

function circularshift!(X::AbstractVector, s::Int)
    n = length(X)
    n == 0 && return X
    s = mod(s, n)
    s == 0 && return X
    if n <= 100
        return circshift_reverse!(X, s)
    elseif sizeof(X) ≥ 32n
        return circshift_perm2!(X, s)
    elseif 8s ≤ n
        return circshift_buf1!(X, s)
    elseif 8(n-s) ≤ n
        return circshift_buf2!(X, s)
    else
        return circshift_reverse!(X, s)
    end
end

# convert `s` to an `Int`, to prevent both type instabilities and
# bad performance if someone is perverse and passes, say, a `BigInt`.
circularshift!(X::AbstractVector, s::Integer) = circularshift!(X, Int(s))

########################################################################
# problem 1(b)

# efficient code to copy a chunk of rows in a matrix.  (These are only called
# internally below when we can guaranteee that the indices are in-bounds.)
function _copyrows!(Xi::Vector, X::AbstractMatrix, chunkrow, i) # X[:,i] to Xi
    chunkrow -= 1
    @simd for j = 1:length(Xi)
        @inbounds Xi[j] = X[chunkrow+j, i]
    end
end
function _copyrows!(X::AbstractMatrix, chunklen, chunkrow, k, i) # X[:,i] to X[:,k]
    @simd for j = chunkrow:chunkrow+chunklen-1
        @inbounds X[j, k] = X[j, i]
    end
end
function _copyrows!(X::AbstractMatrix, chunkrow, i, Xi::Vector) # Xi to X[:,i]
    chunkrow -= 1
    @simd for j = 1:length(Xi)
        @inbounds X[chunkrow+j, i] = Xi[j]
    end
end

# like circshift_perm2! above, but permute the rows of X by s, permuting
# several rows at a time in chunks.  (It would be nicer to avoid the copy-paste
# duplication with some better abstractions here.)
function circshift_perm2!{T}(X::AbstractMatrix{T}, s::Int)
    n = size(X,2)
    n == 0 && return X
    s = mod(s, n)
    s == 0 && return X
    B = BitVector(s)
    ncycles_per_row = gcd(s,n)

    # figure out how big of a chunk we can allocate according to the rules
    Xi = Array{T}(1) # allocate array of 1 element to get size.
    # We need to satisfy:
    #   chunklen*sizeof(Xi) + sizeof(B) ≤ (sizeof(Xi)*size(X,2))÷8 + 128
    chunklen = max(1, size(X,2)÷8 + (128-sizeof(B))÷sizeof(Xi))

    for chunkrow = 1:chunklen:size(X,1) # start of each row chunk
        # the last chunk of rows might be smaller
        chunklen = min(chunklen, size(X,1) - chunkrow + 1)
        resize!(Xi, chunklen)

        fill!(B, false)
        ncycles = ncycles_per_row
        @inbounds for i = 1:s
            if !B[i] # cycle starting at i has not been visited yet
                # Xi = X[i]
                _copyrows!(Xi, X, chunkrow, i)
                j = i
                while true
                    k = j - s + n
                    # X[j] = X[k]
                    _copyrows!(X, chunklen, chunkrow, j, k)
                    B[j] = true
                    j = k
                    while j > s
                        k = j - s # j should be > s
                        # X[j] = X[k]
                        _copyrows!(X, chunklen, chunkrow, j, k)
                        j = k
                    end
                    j == i && break # end of cycle
                end
                k = j + s
                # X[k > n ? k - n : k] = Xi
                _copyrows!(X, chunkrow, k > n ? k - n : k, Xi)
                ncycles -= 1
                ncycles == 0 && break
            end
        end
    end
    return X
end

circularshift!(X::AbstractMatrix, s::Integer) = circshift_perm2!(X, Int(s))

########################################################################
# problem 1(c)

function circshift_bybatch!(X::AbstractMatrix, batchsize::Int)
    m, n = size(X)
    n==0 && return X # we assume n>0 below
    for i = 1:batchsize:m # i is the start of each batch
        b = min(batchsize, m-i+1) # size of the current batch
        s = mod(i, n)
        s′ = s == 0 ? 0 : n - s

        # perform reversals of 1:s′ and s′+1:n for each row j in the batch
        for j = i:i+b-1
            v = view(X, j, :)
            reverse!(v, 1, s′)   # reverse first s′ elements
            reverse!(v, s′+1, n) # reverse last s elements
            s′ -= 1
            if s′ < 0
                s′ += n
            end
        end

        # reverse the whole row for every element in the block at once:
        k = 1
        k′ = n
        @inbounds while k < k′
            @simd for j = i:i+b-1
                t = X[j, k]
                X[j, k] = X[j, k′]
                X[j, k′] = t
            end
            k  += 1
            k′ -= 1
        end
    end
    return X
end

circularshift!(X::AbstractMatrix) = circshift_bybatch!(X, size(X,1))

########################################################################
########################################################################
# problem 2

using StaticArrays

# since we store the spheres in a tree, we also need to keep track of the index
# with which that sphere appeared in the original data, so that we can still return
# that index from findsphere.  We also convert to a floating-point type, since
# otherwise the radius^2 computation can easily overflow.  (Note that initial "slow"
# solution converted to floating-point in the norm computations anyway.)
immutable Sphere{T<:AbstractFloat}
    center::SVector{3,T}
    radius2::T # the radius^2
    index::Int # index (row) of this sphere in the original data
end
function Sphere{T<:Real,S<:Real}(center::AbstractVector{T}, radius::S, i::Integer)
    R = float(promote_type(T, S))
    return Sphere{R}(SVector{3,R}(center), R(radius)^2, Int(i))
end

type KDTree{T<:Real}
    o::Vector{Sphere{T}}
    ix::Int # dimension being sliced (1 to 3, or 0 for leaf nodes)
    x::Float64    # the coordinate of the slice plane
    left::KDTree  # objects ≤ x in coordinate ix
    right::KDTree # objects > x in coordinate ix
    KDTree(o::Vector{Sphere{T}}) = new(o, 0)
    function KDTree(x::Real, ix::Integer, left::KDTree{T}, right::KDTree{T})
        1 ≤ ix ≤ 3 || throw(BoundsError())
        new(Sphere{T}[], ix, x, left, right)
    end
end

function KDTree{T}(o::AbstractVector{Sphere{T}})
    length(o) <= 4 && return KDTree{T}(o)

    # figure out the best dimension ix to divide over,
    # the dividing plane x, and the number (nl,nr) of
    # objects that fall into the left and right subtrees
    ix = 0
    x = zero(T)
    nl = nr = typemax(Int)
    for i = 1:3
        mx = median(map(s -> s.center[i], o))
        mnl = count(s -> s.center[i] - sqrt(s.radius2) ≤ mx, o) # lower bound ≤ mx
        mnr = count(s -> s.center[i] + sqrt(s.radius2) > mx, o) # upper bound > mx
        if max(mnl,mnr) < max(nl,nr)
            ix = i
            x = mx
            nl = mnl
            nr = mnr
        end
    end

    # don't bother subdividing if it doesn't reduce the # of objects much
    4*min(nl,nr) > 3*length(o) && return KDTree{T}(o)

    # create the arrays of objects in each subtree
    ol = Array{Sphere{T}}(nl)
    or = Array{Sphere{T}}(nr)
    il = ir = 0
    for k in eachindex(o)
        s = o[k]
        r = sqrt(s.radius2)
        if s.center[ix] - r ≤ x
            ol[il += 1] = o[k]
        end
        if s.center[ix] + r > x
            or[ir += 1] = o[k]
        end
    end

    return KDTree{T}(x, ix, KDTree(ol), KDTree(or))
end

makespheres_{T<:Real}(data::AbstractMatrix{T}) = [Sphere(data[i,1:3], data[i,4], i) for i = 1:size(data,1)]
makespheres{T<:Real}(data::AbstractMatrix{T}) = KDTree(makespheres_(data))

depth(kd::KDTree) = kd.ix == 0 ? 0 : max(depth(kd.left), depth(kd.right)) + 1

Base.show{T}(io::IO, kd::KDTree{T}) = print(io, "KDTree{$T} of depth ", depth(kd))

function _show(io::IO, kd::KDTree, indent)
    indentstr = " "^indent
    if kd.ix == 0
        println(io, indentstr, length(kd.o), " objects")
    else
        println(io, indentstr, "if x[", kd.ix, "] ≤ ", kd.x, ':')
        _show(io, kd.left, indent + 2)
        println(io, indentstr, "else:")
        _show(io, kd.right, indent + 2)
    end
end

function Base.show(io::IO, ::MIME"text/plain", kd::KDTree)
    println(io, kd, ':')
    _show(io, kd, 0)
end

Base.in{T}(p::SVector{3}, S::Sphere{T}) = sumabs2(p - S.center) ≤ S.radius2

function findsphere{T}(o::AbstractVector{Sphere{T}}, p::SVector{3})
    for i in eachindex(o)
        if p in o[i]
            return o[i].index
        end
    end
    return 0
end

function findsphere{T}(kd::KDTree{T}, p::SVector{3})
    if isempty(kd.o)
        if p[kd.ix] ≤ kd.x
            return findsphere(kd.left, p)
        else
            return findsphere(kd.right, p)
        end
    else
        return findsphere(kd.o, p)
    end
end

findsphere{T}(S::AbstractVector{Sphere{T}}, p::AbstractVector) = findsphere(S, SVector{3}(p))
findsphere{T}(kd::KDTree{T}, p::AbstractVector) = findsphere(kd, SVector{3}(p))
