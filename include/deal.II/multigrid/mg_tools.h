// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2016 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii__mg_tools_h
#define dealii__mg_tools_h

#include <deal.II/base/config.h>
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>

#include <vector>
#include <set>


DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim> class DoFHandler;

/* !@addtogroup mg */
/* @{ */

/**
 * This is a collection of functions operating on, and manipulating the
 * numbers of degrees of freedom in a multilevel triangulation. It is similar
 * in purpose and function to the @p DoFTools namespace, but operates on
 * levels of DoFHandler objects. See there and the documentation of the member
 * functions for more information.
 *
 * @author Wolfgang Bangerth, Guido Kanschat, 1999 - 2005, 2012
 */
namespace MGTools
{
  /**
   * Compute row length vector for multilevel methods.
   */
  template <int dim, int spacedim>
  void
  compute_row_length_vector(const DoFHandler<dim,spacedim> &dofs,
                            const unsigned int level,
                            std::vector<unsigned int> &row_lengths,
                            const DoFTools::Coupling flux_couplings = DoFTools::none);

  /**
   * Compute row length vector for multilevel methods with optimization for
   * block couplings.
   */
  template <int dim, int spacedim>
  void
  compute_row_length_vector(const DoFHandler<dim,spacedim> &dofs,
                            const unsigned int level,
                            std::vector<unsigned int> &row_lengths,
                            const Table<2,DoFTools::Coupling> &couplings,
                            const Table<2,DoFTools::Coupling> &flux_couplings);

  /**
   * Write the sparsity structure of the matrix belonging to the specified @p
   * level. The sparsity pattern is not compressed, so before creating the
   * actual matrix you have to compress the matrix yourself, using
   * <tt>SparseMatrixStruct::compress()</tt>.
   *
   * There is no need to consider hanging nodes here, since only one level is
   * considered.
   */
  template <typename DoFHandlerType, typename SparsityPatternType>
  void
  make_sparsity_pattern (const DoFHandlerType &dof_handler,
                         SparsityPatternType  &sparsity,
                         const unsigned int    level);

  /**
   * Make a sparsity pattern including fluxes of discontinuous Galerkin
   * methods.
   * @see
   * @ref make_sparsity_pattern
   * and
   * @ref DoFTools
   */
  template <int dim, typename SparsityPatternType, int spacedim>
  void
  make_flux_sparsity_pattern (const DoFHandler<dim,spacedim> &dof_handler,
                              SparsityPatternType            &sparsity,
                              const unsigned int              level);

  /**
   * Create sparsity pattern for the fluxes at refinement edges. The matrix
   * maps a function of the fine level space @p level to the coarser space.
   *
   * make_flux_sparsity_pattern()
   */
  template <int dim, typename SparsityPatternType, int spacedim>
  void
  make_flux_sparsity_pattern_edge (const DoFHandler<dim,spacedim> &dof_handler,
                                   SparsityPatternType            &sparsity,
                                   const unsigned int              level);
  /**
   * This function does the same as the other with the same name, but it gets
   * two additional coefficient matrices. A matrix entry will only be
   * generated for two basis functions, if there is a non-zero entry linking
   * their associated components in the coefficient matrix.
   *
   * There is one matrix for couplings in a cell and one for the couplings
   * occurring in fluxes.
   */
  template <int dim, typename SparsityPatternType, int spacedim>
  void
  make_flux_sparsity_pattern (const DoFHandler<dim,spacedim>    &dof,
                              SparsityPatternType               &sparsity,
                              const unsigned int                 level,
                              const Table<2,DoFTools::Coupling> &int_mask,
                              const Table<2,DoFTools::Coupling> &flux_mask);

  /**
   * Create sparsity pattern for the fluxes at refinement edges. The matrix
   * maps a function of the fine level space @p level to the coarser space.
   * This is the version restricting the pattern to the elements actually
   * needed.
   *
   * make_flux_sparsity_pattern()
   */
  template <int dim, typename SparsityPatternType, int spacedim>
  void
  make_flux_sparsity_pattern_edge (const DoFHandler<dim,spacedim>    &dof_handler,
                                   SparsityPatternType               &sparsity,
                                   const unsigned int                 level,
                                   const Table<2,DoFTools::Coupling> &flux_mask);

  /**
   * Count the dofs block-wise on each level.
   *
   * Result is a vector containing for each level a vector containing the
   * number of dofs for each block (access is <tt>result[level][block]</tt>).
   */
  template <typename DoFHandlerType>
  void
  count_dofs_per_block (const DoFHandlerType                               &dof_handler,
                        std::vector<std::vector<types::global_dof_index> > &dofs_per_block,
                        std::vector<unsigned int> target_block = std::vector<unsigned int>());

  /**
   * Count the dofs component-wise on each level.
   *
   * Result is a vector containing for each level a vector containing the
   * number of dofs for each component (access is
   * <tt>result[level][component]</tt>).
   */
  template <int dim, int spacedim>
  void
  count_dofs_per_component (const DoFHandler<dim,spacedim> &mg_dof,
                            std::vector<std::vector<types::global_dof_index> > &result,
                            const bool only_once = false,
                            std::vector<unsigned int> target_component = std::vector<unsigned int>());

  /**
   * Generate a list of those degrees of freedom at the boundary of the domain
   * that should be eliminated from the matrix because they will be
   * constrained by Dirichlet boundary conditions.
   *
   * This is the multilevel equivalent of
   * VectorTools::interpolate_boundary_values, but since the multilevel method
   * does not have its own right hand side, the function values returned by
   * the function object that is part of the function_map argument are
   * ignored.
   *
   * @arg <tt>boundary_indices</tt> is a vector which on return contains all
   * indices of degrees of freedom for each level that are at the part of the
   * boundary identified by the function_map argument. Its length has to match
   * the number of levels in the dof handler object.
   *
   * Previous content in @p boundary_indices is not overwritten,
   * but added to.
   */
  template <int dim, int spacedim>
  void
  make_boundary_list (const DoFHandler<dim,spacedim>      &mg_dof,
                      const typename FunctionMap<dim>::type &function_map,
                      std::vector<std::set<types::global_dof_index> > &boundary_indices,
                      const ComponentMask                   &component_mask = ComponentMask());

  /**
   * The same function as above, but return an IndexSet rather than a
   * std::set<unsigned int> on each level.
   *
   * Previous content in @p boundary_indices is not overwritten,
   * but added to.
   */
  template <int dim, int spacedim>
  void
  make_boundary_list (const DoFHandler<dim,spacedim>      &mg_dof,
                      const typename FunctionMap<dim>::type &function_map,
                      std::vector<IndexSet>                 &boundary_indices,
                      const ComponentMask               &component_mask = ComponentMask());

  /**
   * The same function as above, but return an IndexSet rather than a
   * std::set<unsigned int> on each level and use a std::set of boundary_ids
   * as input.
   *
   * Previous content in @p boundary_indices is not overwritten, but added to.
   */
  template <int dim, int spacedim>
  void
  make_boundary_list (const DoFHandler<dim,spacedim>      &mg_dof,
                      const std::set<types::boundary_id> &boundary_ids,
                      std::vector<IndexSet>                 &boundary_indices,
                      const ComponentMask               &component_mask = ComponentMask());


  /**
   * Generate constraints to be used for assembling multigrid matrices
   * for locally refined meshes.
   *
   * Equivalent to DoFTools::make_hanging_nodes_constraints() for
   * level dofs. Note that albeit we run the local smoother in the
   * interior of the refined part of the mesh, that is, using
   * essential boundary conditions on the refinement edge, this
   * function eliminates actual hanging nodes, such that a basis for
   * the coarse grid functions onthe edge is still available. This
   * structure is needed not for smoothing, but for the additional
   * edge matrices used for computing global residuals after local
   * smoothing. See Multigrid::set_edge_matrices() and
   * Multigrid::set_edge_flux_matrices() for using these matrices and
   * Meshworker::Assembler::MGMatrixSimple for a way to build them.
   *
   * While this function emulated
   * DoFTools::make_hanging_nodes_constraints(), there is a major
   * difference here. The standard DoFHandler as well as the
   * hp::DoFHandler double degrees of freedom on refinement edges,
   * such that we can constrain all degrees of freedom on the finer
   * side and replace them by linear combinations on the coarse
   * side. With level matrices, we do not have this option, since
   * degrees of freedom on the coarser cell belong to a different
   * level. Instead, we have to constrain against degrees of freedom
   * within the fine cells. Since we have to write constraints
   * explicitly, we have to select some more or less randomly for
   * elimination by others. A few criteria apply though:
   *
   * <ol>
   * <li> Degrees of freedom on the vertices bounding the parent face are not
   * constrained.</li>
   * <li> In three dimensions, degrees of freedom on the edges bounding the
   * parent face are constrained by degrees of freedom on the same edge.</li>
   * <li> All degrees of freedom on interior boundaries of the refinement patch
   * are constrained.</li>
   * <li> With all this, we have to make sure that the whole polynomial space on
   * the parent face is contained in the constrained space. Therefore, if
   * interpolation is defined by moments of increasing order, the highest order
   * must be present.</li>
   * </ol>
   *
   * As a first option, we can choose as degrees of freedom not fixed
   * by rules 1 and 2 all interior degrees of freedom on the face of
   * child one.  But then we run into the extrapolation trap. The
   * polynomials grow very fast on the neighboring cells and thus the
   * whole procedure becomes numerically unstable.
   *
   * The operation of contraining degrees of freedom is an embedding
   * of the coarse space on the interface into the fine space, such
   * that the eliminated degrees of freedom in the end span the
   * complement of the range of this embedding. This is the standard
   * elimination in DoFTools and the second part of the mapping needed
   * here. But now we need an endomorphism with the chosen hanging
   * nodes spanning the kernel. Thus, we first need a mapping from the
   * fine space into the coarse space, which is then embedded back.
   *
   * A natural choice for this mapping is the adjoint of the
   * embedding, since its kernel is spanned by the eliminated degrees
   * of freedom.
   */
  template <typename DoFHandlerType>
  void
  make_hanging_node_constraints (const DoFHandlerType &dof_handler,
                                 MGLevelObject<ConstraintMatrix> &constraints);

  
  /**
   * For each level in a multigrid hierarchy, produce an IndexSet that
   * indicates which of the degrees of freedom are along interfaces of this
   * level to cells that only exist on coarser levels.
   */
  template <int dim, int spacedim>
  void
  extract_inner_interface_dofs (const DoFHandler<dim,spacedim> &mg_dof_handler,
                                std::vector<IndexSet>  &coarser_edge_dofs,
                                std::vector<IndexSet>  &finer_edge_dofs);


  template <int dim, int spacedim>
  void
  extract_non_interface_dofs (const DoFHandler<dim,spacedim> &mg_dof_handler,
                              std::vector<std::set<types::global_dof_index> > &non_interface_dofs);
}

/* @} */

DEAL_II_NAMESPACE_CLOSE

#endif
