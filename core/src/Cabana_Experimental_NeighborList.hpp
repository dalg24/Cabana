/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_EXPERIMENTAL_NEIGHBOR_LIST_HPP
#define CABANA_EXPERIMENTAL_NEIGHBOR_LIST_HPP

#include <Cabana_NeighborList.hpp>
#include <Cabana_Slice.hpp>

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace Cabana
{
namespace stdcxx20
{
template <class T>
struct remove_cvref
{
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
} // namespace stdcxx20

namespace Impl
{
template <typename Slice,
          typename = std::enable_if_t<Cabana::is_slice<Slice>::value>>
struct SliceAndRadius : Slice
{
    using value_type = typename Slice::value_type;
    value_type radius;
    SliceAndRadius( Slice slice, value_type r )
        : Slice( std::move( slice ) )
        , radius( r )
    {
    }
};

template <typename Slice, typename = std::enable_if_t<Cabana::is_slice<
                              std::remove_reference_t<Slice>>::value>>
auto makePredicates(
    Slice&& slice, typename stdcxx20::remove_cvref_t<Slice>::value_type radius )
{
    return SliceAndRadius<stdcxx20::remove_cvref_t<Slice>>{
        std::forward<Slice>( slice ), radius };
}
} // namespace Impl
} // namespace Cabana

template <typename Slice>
struct ArborX::AccessTraits<Slice, ArborX::PrimitivesTag,
                            std::enable_if_t<Cabana::is_slice<Slice>{}>>
{
    using memory_space = typename Slice::memory_space;
    using size_type = typename Slice::size_type;
    static KOKKOS_FUNCTION size_type size( Slice const& x ) { return x.size(); }
    static KOKKOS_FUNCTION Point get( Slice const& x, size_type i )
    {
        return { static_cast<float>( x( i, 0 ) ),
                 static_cast<float>( x( i, 1 ) ),
                 static_cast<float>( x( i, 2 ) ) };
    }
};

template <typename SliceLike>
struct ArborX::AccessTraits<SliceLike, ArborX::PredicatesTag>
{
    using memory_space = typename SliceLike::memory_space;
    using size_type = typename SliceLike::size_type;
    static KOKKOS_FUNCTION size_type size( SliceLike const& x )
    {
        return x.size();
    }
    static KOKKOS_FUNCTION auto get( SliceLike const& x, size_type i )
    {
        assert( i < size( x ) );
        auto const point =
            AccessTraits<typename SliceLike::slice_type, PrimitivesTag>::get(
                x, i );
        return attach( intersects( Sphere{ point, x.radius } ), (int)i );
    }
};

namespace Cabana
{
namespace Impl
{

template <typename Tag>
struct CollisionFilter;

template <>
struct CollisionFilter<FullNeighborTag>
{
    KOKKOS_FUNCTION bool static keep( int i, int j ) noexcept
    {
        return i != j; // discard self-collision
    }
};

template <>
struct CollisionFilter<HalfNeighborTag>
{
    KOKKOS_FUNCTION static bool keep( int i, int j ) noexcept { return i > j; }
};

// Custom callback for ArborX::BVH::query()
template <typename Tag>
struct NeighborDiscriminatorCallback
{
    template <typename Predicate, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index,
                                     OutputFunctor const& out ) const
    {
        int const predicate_index = getData( predicate );
        if ( CollisionFilter<Tag>::keep( predicate_index, primitive_index ) )
        {
            out( primitive_index );
        }
    }
};

// Count in the first pass
template <typename Counts, typename Tag>
struct NeighborDiscriminatorCallback2D_FirstPass
{
    using TreeTraversalControl = ArborX::Experimental::TreeTraversalQuick;
    Counts counts;
    template <typename Predicate>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index ) const
    {
        int const predicate_index = getData( predicate );
        ++counts( predicate_index ); // WARNING see below**
        if ( std::is_same<Tag, FullNeighborTag>::value )
            ++counts( primitive_index );
    }
};

// Preallocate and attempt fill in the first pass
template <typename Counts, typename Neighbors, typename Tag>
struct NeighborDiscriminatorCallback2D_FirstPass_BufferOptimization
{
    using TreeTraversalControl = ArborX::Experimental::TreeTraversalQuick;
    Counts counts;
    Neighbors neighbors;
    template <typename Predicate>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index ) const
    {
        int const predicate_index = getData( predicate );
        if ( counts( predicate_index ) < neighbors.extent_int( 1 ) &&
             ( !std::is_same<Tag, FullNeighborTag>::value ||
               counts( primitive_index ) < neighbors.extent_int( 1 ) ) )
        {
            neighbors( predicate_index, counts( predicate_index )++ ) =
                primitive_index; // WARNING see below*
            if ( std::is_same<Tag, FullNeighborTag>::value )
                neighbors( primitive_index, counts( primitive_index )++ ) =
                    predicate_index;
        }
        else
        {
            ++counts( predicate_index );
            if ( std::is_same<Tag, FullNeighborTag>::value )
                ++counts( primitive_index );
        }
    }
};

// Fill in the second pass
template <typename Counts, typename Neighbors, typename Tag>
struct NeighborDiscriminatorCallback2D_SecondPass
{
    using TreeTraversalControl = ArborX::Experimental::TreeTraversalQuick;
    Counts counts;
    Neighbors neighbors;
    template <typename Predicate>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index ) const
    {
        int const predicate_index = getData( predicate );
        assert( counts( predicate_index ) < neighbors.extent_int( 1 ) );
        assert( ( !std::is_same<Tag, FullNeighborTag>::value ||
                  counts( primitive_index ) < neighbors.extent_int( 1 ) ) );
        neighbors( predicate_index, counts( predicate_index )++ ) =
            primitive_index; // WARNING see below**
        if ( std::is_same<Tag, FullNeighborTag>::value )
            neighbors( primitive_index, counts( primitive_index )++ ) =
                predicate_index;
    }
};

// NOTE** Taking advantage of the knowledge that one predicate is processed by a
// single thread.  Count increment should be atomic otherwise.

} // namespace Impl

namespace Experimental
{
template <typename MemorySpace, typename Tag>
struct CrsGraph
{
    Kokkos::View<int*, MemorySpace> col_ind;
    Kokkos::View<int*, MemorySpace> row_ptr;
    typename MemorySpace::size_type first;
    typename MemorySpace::size_type last;
};

template <typename DeviceType, typename Slice, typename Tag>
auto makeNeighborList( Tag, Slice const& coordinate_slice,
                       typename Slice::size_type first,
                       typename Slice::size_type last,
                       typename Slice::value_type radius, int buffer_size = 0 )
{
    assert( buffer_size >= 0 );

    using MemorySpace = typename DeviceType::memory_space;
    using ExecutionSpace = typename DeviceType::execution_space;
    ExecutionSpace space{};

    ArborX::BVH<MemorySpace> bvh( space, coordinate_slice );

    Kokkos::View<int*, DeviceType> indices(
        Kokkos::view_alloc( "indices", Kokkos::WithoutInitializing ), 0 );
    Kokkos::View<int*, DeviceType> offset(
        Kokkos::view_alloc( "offset", Kokkos::WithoutInitializing ), 0 );
    bvh.query(
        space, Impl::makePredicates( coordinate_slice, radius ),
        Impl::NeighborDiscriminatorCallback<Tag>{}, indices, offset,
        ArborX::Experimental::TraversalPolicy().setBufferSize( buffer_size ) );

    return CrsGraph<MemorySpace, Tag>{ std::move( indices ),
                                       std::move( offset ), first, last };
}

template <typename MemorySpace, typename Tag>
struct Dense
{
    Kokkos::View<int*, MemorySpace> cnt;
    Kokkos::View<int**, MemorySpace> val;
    typename MemorySpace::size_type first;
    typename MemorySpace::size_type last;
};

template <typename DeviceType, typename Slice, typename Tag>
auto make2DNeighborList( Tag, Slice const& coordinate_slice,
                         typename Slice::size_type first,
                         typename Slice::size_type last,
                         typename Slice::value_type radius,
                         int buffer_size = 0 )
{
    assert( buffer_size >= 0 );

    using MemorySpace = typename DeviceType::memory_space;
    using ExecutionSpace = typename DeviceType::execution_space;
    ExecutionSpace space{};

    ArborX::BVH<MemorySpace> bvh( space, coordinate_slice );

    auto const predicates = Impl::makePredicates( coordinate_slice, radius );

    auto const n_queries =
        ArborX::AccessTraits<decltype( predicates ),
                             ArborX::PredicatesTag>::size( predicates );

    Kokkos::View<int**, DeviceType> neighbors;
    Kokkos::View<int*, DeviceType> counts( "counts", n_queries );
    if ( buffer_size > 0 )
    {
        neighbors = Kokkos::View<int**, DeviceType>(
            Kokkos::view_alloc( "neighbors", Kokkos::WithoutInitializing ),
            n_queries, buffer_size );
        bvh.query(
            space, predicates,
            Impl::NeighborDiscriminatorCallback2D_FirstPass_BufferOptimization<
                decltype( counts ), decltype( neighbors ), Tag>{ counts,
                                                                 neighbors } );
    }
    else
    {
        bvh.query(
            space, predicates,
            Impl::NeighborDiscriminatorCallback2D_FirstPass<decltype( counts ),
                                                            Tag>{ counts } );
    }

    auto const max_neighbors = ArborX::max( space, counts );
    if ( max_neighbors <= buffer_size )
    {
        // NOTE We do not bother shrinking to eliminate the excess allocation.
        // NOTE If buffer_size is 0, neighbors is default constructed.  This is
        // fine with the current design/implementation of NeighborList access
        // traits.
        return Dense<MemorySpace, Tag>{ counts, neighbors, first, last };
    }

    neighbors = Kokkos::View<int**, DeviceType>(
        Kokkos::view_alloc( "neighbors", Kokkos::WithoutInitializing ),
        n_queries, max_neighbors ); // realloc storage for neighbors
    Kokkos::deep_copy( counts, 0 ); // reset counts to zero
    bvh.query( space, predicates,
               Impl::NeighborDiscriminatorCallback2D_SecondPass<
                   decltype( counts ), decltype( neighbors ), Tag>{
                   counts, neighbors } );

    return Dense<MemorySpace, Tag>{ counts, neighbors, first, last };
}

} // namespace Experimental

template <typename MemorySpace, typename Tag>
class NeighborList<Experimental::CrsGraph<MemorySpace, Tag>>
{
    using size_type = std::size_t;
    using crs_graph_type = Experimental::CrsGraph<MemorySpace, Tag>;

  public:
    using memory_space = MemorySpace;
    static KOKKOS_FUNCTION size_type
    numNeighbor( crs_graph_type const& crs_graph, size_type p )
    {
        // assert( (int)p >= 0 && p < crs_graph.total );
        if ( p < crs_graph.first || p >= crs_graph.last )
            return 0;
        return crs_graph.row_ptr( p + 1 ) - crs_graph.row_ptr( p );
    }
    static KOKKOS_FUNCTION size_type
    getNeighbor( crs_graph_type const& crs_graph, size_type p, size_type n )
    {
        assert( n < numNeighbor( crs_graph, p ) );
        return crs_graph.col_ind( crs_graph.row_ptr( p ) + n );
    }
};

template <typename MemorySpace, typename Tag>
class NeighborList<Experimental::Dense<MemorySpace, Tag>>
{
    using size_type = std::size_t;
    using specialization_type = Experimental::Dense<MemorySpace, Tag>;

  public:
    using memory_space = MemorySpace;
    static KOKKOS_FUNCTION size_type numNeighbor( specialization_type const& d,
                                                  size_type p )
    {
        // assert( (int)p >= 0 && p < d.total );
        if ( p < d.first || p >= d.last )
            return 0;
        return d.cnt( p );
    }
    static KOKKOS_FUNCTION size_type getNeighbor( specialization_type const& d,
                                                  size_type p, size_type n )
    {
        assert( n < numNeighbor( d, p ) );
        return d.val( p, n );
    }
};

} // namespace Cabana

#endif
