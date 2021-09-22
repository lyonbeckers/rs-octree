#![deny(clippy::pedantic)]
#![deny(clippy::perf)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

pub mod error;
#[cfg(test)]
mod test;

use std::{
    fmt::{self, Debug},
    iter::FromIterator,
    sync::Arc,
};

use crate::error::{Error, Result};
use aabb::{vector_abs, Aabb, NumTraits};
use nalgebra::{Scalar, Vector3};
use num::{traits::Bounded, NumCast};
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub static DEFAULT_MAX: usize = 32;

pub trait PointData<N: Scalar>: Copy {
    fn get_point(&self) -> Vector3<N>;
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Copy, Clone)]
#[allow(dead_code)]
enum Paternity {
    ProudParent,
    ChildFree,
}

pub struct OctreeIter<N: Scalar, T: PointData<N>> {
    elements: std::vec::IntoIter<T>,
    phantom: std::marker::PhantomData<N>,
}

impl<'a, N: Scalar, T: PointData<N>> Iterator for OctreeIter<N, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        match self.elements.next() {
            Some(r) => Some(r),
            None => None,
        }
    }
}

impl<N, T> IntoIterator for Octree<N, T>
where
    N: NumTraits + Sync + Send + Copy + Clone + Serialize + DeserializeOwned,
    T: PointData<N> + PartialEq + Debug + Sync + Send,
{
    type Item = T;
    type IntoIter = OctreeIter<N, T>;
    fn into_iter(self) -> Self::IntoIter {
        let aabb = self.aabb;
        OctreeIter {
            elements: self.query_range(aabb).into_iter(),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<N, T> FromIterator<T> for Octree<N, T>
where
    N: Sync + Send + Bounded + NumTraits + Copy + Clone + Serialize + DeserializeOwned,
    T: PointData<N> + PartialEq + Debug + Sync + Send,
{
    fn from_iter<A: IntoIterator<Item = T>>(iter: A) -> Self {
        let mut smallest = Vector3::<N>::new(
            Bounded::max_value(),
            Bounded::max_value(),
            Bounded::max_value(),
        );
        let mut largest = Vector3::<N>::new(
            Bounded::min_value(),
            Bounded::min_value(),
            Bounded::min_value(),
        );

        let items = iter.into_iter().collect::<Vec<T>>();

        if items.is_empty() {
            smallest = Vector3::zeros();
            largest = Vector3::zeros();
        } else {
            for item in &items {
                let pt = item.get_point();

                smallest.x = pt.x.min(smallest.x);
                smallest.y = pt.y.min(smallest.y);
                smallest.z = pt.z.min(smallest.z);

                largest.x = pt.x.max(largest.x);
                largest.y = pt.y.max(largest.y);
                largest.z = pt.z.max(largest.z);
            }
        }

        let mut octree = Octree::new(Aabb::from_extents(smallest, largest), DEFAULT_MAX);

        for item in &items {
            octree.insert(*item).unwrap();
        }

        octree
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct Octree<N: Scalar, T: PointData<N>> {
    aabb: Aabb<N>,
    max_elements: usize,
    elements: Vec<T>,
    children: Vec<Arc<RwLock<Octree<N, T>>>>,
    paternity: Paternity,
}

#[allow(dead_code)]
impl<N, T> Octree<N, T>
where
    N: Sync + Send + NumTraits + Copy + Clone,
    T: PointData<N> + PartialEq + Debug + Sync + Send,
{
    pub fn new(aabb: Aabb<N>, max_elements: usize) -> Octree<N, T> {
        tracing::debug!(target: "creating octree", min = ?aabb.get_min(), max = ?aabb.get_max());

        Octree {
            aabb,
            max_elements,
            elements: Vec::with_capacity(max_elements),
            children: Vec::with_capacity(8),
            paternity: Paternity::ChildFree,
        }
    }

    pub fn get_aabb(&self) -> Aabb<N> {
        self.aabb
    }

    pub fn get_max_elements(&self) -> usize {
        self.max_elements
    }

    /// # Errors
    /// will panic if the subdivision does not have the correct dimensions
    #[allow(clippy::too_many_lines)]
    fn subdivide(&mut self) -> Result<(), N> {
        let zero: N = NumCast::from(0).unwrap();
        let one: N = NumCast::from(1).unwrap();
        let two: N = NumCast::from(2).unwrap();

        // Hacky way of checking if it's an integer and then adjusting min so all values behave like indices
        let adj = if one / two == zero { one } else { zero };

        let min = self.aabb.get_min();
        let max = self.aabb.get_max();

        let dimensions = vector_abs(self.aabb.dimensions);

        let smaller_half = dimensions / two;
        let larger_half = dimensions - smaller_half - Vector3::new(adj, adj, adj);

        let (tx, rx) = crossbeam_channel::unbounded::<Octree<N, T>>();

        let max_elements = Arc::new(self.max_elements);

        rayon::scope(move |s| {
            //down back left
            let sub_max = min + larger_half;

            let downbackleft = Aabb::<N>::from_extents(min, sub_max);
            tx.send(Octree::new(downbackleft, *max_elements)).unwrap();

            if dimensions.x > one {
                let tx1 = tx.clone();
                let max_elements = Arc::clone(&max_elements);

                s.spawn(move |s| {
                    let sub_min = min + Vector3::new(larger_half.x + adj, zero, zero);
                    let sub_max =
                        Vector3::new(max.x, sub_min.y + larger_half.y, sub_min.z + larger_half.z);

                    let downbackright = Aabb::<N>::from_extents(sub_min, sub_max);
                    tx1.send(Octree::new(downbackright, *max_elements)).unwrap();

                    if dimensions.z > one {
                        s.spawn(move |s| {
                            //down forward right
                            let sub_min =
                                min + Vector3::new(larger_half.x + adj, zero, larger_half.z + adj);
                            let sub_max = Vector3::new(max.x, sub_min.y + larger_half.y, max.z);
                            let downforwardright = Aabb::<N>::from_extents(sub_min, sub_max);
                            tx1.send(Octree::new(downforwardright, *max_elements))
                                .unwrap();

                            if dimensions.y > one {
                                s.spawn(move |_| {
                                    //up forward right
                                    let sub_min = min
                                        + Vector3::new(
                                            larger_half.x + adj,
                                            larger_half.y + adj,
                                            larger_half.z + adj,
                                        );
                                    let upforwardright = Aabb::<N>::from_extents(sub_min, max);
                                    tx1.send(Octree::new(upforwardright, *max_elements))
                                        .unwrap();
                                });
                            }
                        });
                    }
                });
            }

            if dimensions.z > one {
                let tx2 = tx.clone();
                let max_elements = Arc::clone(&max_elements);

                s.spawn(move |s| {
                    //down forward left
                    let sub_min = min + Vector3::new(zero, zero, larger_half.z + adj);
                    let sub_max =
                        Vector3::new(sub_min.x + larger_half.x, sub_min.y + larger_half.y, max.z);

                    let downforwardleft = Aabb::<N>::from_extents(sub_min, sub_max);
                    tx2.send(Octree::new(downforwardleft, *max_elements))
                        .unwrap();

                    if dimensions.y > one {
                        s.spawn(move |_| {
                            //up forward left
                            let sub_min =
                                min + Vector3::new(zero, larger_half.y + adj, larger_half.z + adj);
                            let sub_max = Vector3::new(sub_min.x + larger_half.x, max.y, max.z);
                            let upforwardleft = Aabb::<N>::from_extents(sub_min, sub_max);
                            tx2.send(Octree::new(upforwardleft, *max_elements)).unwrap();
                        });
                    }
                });
            }

            if dimensions.y > one {
                let tx3 = tx.clone();
                let max_elements = Arc::clone(&max_elements);

                s.spawn(move |s| {
                    //up back left
                    let sub_min = min + Vector3::new(zero, larger_half.y + adj, zero);
                    let sub_max =
                        Vector3::new(sub_min.x + larger_half.x, max.y, sub_min.z + larger_half.z);
                    let upbackleft = Aabb::<N>::from_extents(sub_min, sub_max);
                    tx3.send(Octree::new(upbackleft, *max_elements)).unwrap();

                    if dimensions.x > one {
                        s.spawn(move |_| {
                            //up back right
                            let sub_min =
                                min + Vector3::new(larger_half.x + adj, larger_half.y + adj, zero);
                            let sub_max = Vector3::new(max.x, max.y, sub_min.z + larger_half.z);
                            let upbackright = Aabb::<N>::from_extents(sub_min, sub_max);
                            tx3.send(Octree::new(upbackright, *max_elements)).unwrap();
                        });
                    }
                });
            }
        });

        for received in rx {
            self.children.push(Arc::new(RwLock::new(received)));
        }

        self.paternity = Paternity::ProudParent;

        let total_volume = self
            .children
            .par_iter()
            .map(|child| {
                let w = child.write();
                w.aabb.dimensions.x * w.aabb.dimensions.y * w.aabb.dimensions.z
            })
            .sum();

        let volume = dimensions.x * dimensions.y * dimensions.z;

        if cfg!(debug_assertions) {
            if total_volume == volume {
                Ok(())
            } else {
                Err(SubdivisionError {
                    error_type: SubdivisionErrorType::IncorrectDimensions(total_volume, volume),
                }
                .into())
            }
        } else {
            Ok(())
        }
    }

    /// Removes the element at the point
    pub fn remove_item(&mut self, point: Vector3<N>) {
        if let Paternity::ChildFree = self.paternity {
            if self.elements.is_empty() {
                return;
            }
        }

        self.elements = self
            .elements
            .par_iter()
            .filter(|element| element.get_point() != point)
            .copied()
            .collect();

        if let Paternity::ProudParent = self.paternity {
            self.children.par_iter_mut().for_each(|child| {
                child.write().remove_item(point);
            });
        }
    }

    /// Removes all elements which fit inside range, silently avoiding positions that do not fit inside the octree
    pub fn remove_range(&mut self, range: Aabb<N>) {
        if let Paternity::ChildFree = self.paternity {
            if self.elements.is_empty() {
                return;
            }
        }

        self.elements = self
            .elements
            .par_iter()
            .filter(|element| !range.contains_point(element.get_point()))
            .copied()
            .collect();

        if let Paternity::ProudParent = self.paternity {
            self.children.par_iter_mut().for_each(|child| {
                child.write().remove_range(range);
            });
        }
    }

    pub fn insert_elements(&mut self, elements: Vec<T>) -> Result<(), N> {
        let mut elements = elements
            .into_par_iter()
            .filter(|element| self.aabb.contains_point(element.get_point()))
            .collect::<Vec<T>>();

        if elements.is_empty() {
            return Err(Error::InsertionError(InsertionError {
                error_type: InsertionErrorType::OutOfBounds(self.aabb),
            }));
        }

        let available = self.max_elements - self.elements.len();

        //overwrite duplicates
        self.elements.par_iter_mut().for_each(|element| {
            if let Some(dupe) = elements
                .iter()
                .find(|inc| element.get_point() == inc.get_point())
            {
                *element = *dupe
            }
        });

        //cull out duplicates
        elements = elements
            .par_iter()
            .filter(|inc| {
                !self
                    .elements
                    .iter()
                    .any(|orig| orig.get_point() == inc.get_point())
            })
            .copied()
            .collect::<Vec<T>>();

        let remaining = elements.split_off(available.min(elements.len()));

        match &self.paternity {
            Paternity::ChildFree | Paternity::ProudParent
                if self.max_elements > self.elements.len() =>
            {
                self.elements.par_extend(elements);

                if self.paternity == Paternity::ChildFree
                    && self.elements.len() == self.max_elements
                {
                    self.subdivide()?
                }

                if self.elements.len() > self.max_elements {
                    return Err(Error::InsertionError(InsertionError {
                        error_type: InsertionErrorType::Overflow,
                    }));
                }
            }
            Paternity::ChildFree => self.subdivide()?,

            _ => {}
        }

        if remaining.is_empty() {
            return Ok(());
        }

        match &self.paternity {
            Paternity::ProudParent => {
                let (tx, rx) = crossbeam_channel::unbounded::<Result<(), N>>();

                self.children.par_iter_mut().for_each_with(
                    (tx, remaining),
                    |(tx, remaining), child| {
                        match child.write().insert_elements(remaining.clone()) {
                            Ok(_) => tx.send(Ok(())),
                            Err(err) => tx.send(Err(err)),
                        }
                        .unwrap();
                    },
                );

                let mut received = rx.into_iter();
                if let Some(r) = received.find(Result::is_ok) {
                    return r;
                } else if let Some(r) = received.find(Result::is_ok) {
                    return r;
                }

                Err(Error::<N>::InsertionError(InsertionError {
                    error_type: InsertionErrorType::BlockFull(self.aabb),
                }))
            }
            _ => Err(Error::<N>::InsertionError(InsertionError {
                error_type: InsertionErrorType::Empty,
            })),
        }
    }

    pub fn insert(&mut self, element: T) -> Result<(), N> {
        let pt = element.get_point();

        if !self.aabb.contains_point(pt) {
            return Err(Error::<N>::InsertionError(InsertionError {
                error_type: InsertionErrorType::OutOfBounds(self.aabb),
            }));
        }

        //if element already exists at point, replace it
        if let Some(dupe_element) = self
            .elements
            .clone()
            .par_iter_mut()
            .find_any(|element| element.get_point() == pt)
        {
            *dupe_element = element;
            return Ok(());
        }

        //do first match because you still need to insert into children after subdividing, not either/or
        match &self.paternity {
            Paternity::ChildFree | Paternity::ProudParent
                if self.max_elements > self.elements.len() =>
            {
                self.elements.push(element);

                return Ok(());
            }

            Paternity::ChildFree => self.subdivide()?,
            _ => {}
        }

        match &self.paternity {
            Paternity::ProudParent => {
                let (tx, rx) = crossbeam_channel::unbounded::<Result<(), N>>();

                self.children.par_iter_mut().for_each_with(tx, |tx, child| {
                    match child.write().insert(element) {
                        Ok(_) => tx.send(Ok(())),
                        Err(err) => tx.send(Err(err)),
                    }
                    .unwrap();
                });

                let mut received = rx.into_iter();
                if let Some(r) = received.find(Result::is_ok) {
                    return r;
                } else if let Some(r) = received.find(Result::is_ok) {
                    return r;
                }

                Err(Error::<N>::InsertionError(InsertionError {
                    error_type: InsertionErrorType::BlockFull(self.aabb),
                }))
            }

            _ => Err(Error::<N>::InsertionError(InsertionError {
                error_type: InsertionErrorType::Empty,
            })),
        }
    }

    pub fn count(&self) -> usize {
        let mut count: usize = self.elements.len();

        match &self.paternity {
            Paternity::ChildFree => count,
            Paternity::ProudParent => {
                for child in &self.children {
                    count += child.read().count();
                }
                count
            }
        }
    }

    pub fn query_point(&self, point: Vector3<N>) -> Option<T> {
        if !self.aabb.contains_point(point) {
            return None;
        }

        if let Some(found) = self
            .elements
            .par_iter()
            .find_any(|element| element.get_point() == point)
            .copied()
        {
            return Some(found);
        }

        if let Paternity::ChildFree = self.paternity {
            return None;
        }

        let (tx, rx) = crossbeam_channel::unbounded::<T>();

        self.children.par_iter().for_each_with(tx, |tx, child| {
            if let Some(result) = child.read().query_point(point) {
                tx.send(result).unwrap();
            }
        });

        rx.into_iter().next()
    }

    pub fn query_range(&self, range: Aabb<N>) -> Vec<T> {
        let mut elements_in_range: Vec<T> = Vec::with_capacity(self.max_elements);

        if !self.aabb.intersects_bounds(range) {
            return elements_in_range;
        }

        if let Paternity::ChildFree = self.paternity {
            if self.elements.is_empty() {
                return elements_in_range;
            }
        }

        let (tx, rx) = crossbeam_channel::unbounded::<T>();

        self.elements.par_iter().for_each_with(tx, |tx, element| {
            if range.contains_point(element.get_point()) {
                tx.send(*element).unwrap();
            }
        });

        elements_in_range.extend(rx.into_iter());

        if let Paternity::ChildFree = self.paternity {
            return elements_in_range;
        }

        let (tx, rx) = crossbeam_channel::unbounded::<Vec<T>>();

        self.children.par_iter().for_each_with(tx, |tx, child| {
            tx.send(child.read().query_range(range)).unwrap();
        });

        for mut received in rx {
            elements_in_range.append(&mut received)
        }

        elements_in_range
    }
}

#[derive(Clone, Debug)]
pub enum SubdivisionErrorType<N: Scalar> {
    IncorrectDimensions(N, N),
}

#[derive(Debug, Clone)]
pub struct SubdivisionError<N: Scalar> {
    error_type: SubdivisionErrorType<N>,
}

impl<N: Scalar> fmt::Display for SubdivisionError<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.error_type)
    }
}

impl<N: Scalar> std::error::Error for SubdivisionError<N> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[derive(Clone, Debug)]
pub enum InsertionErrorType<N: Scalar> {
    Empty,
    Overflow,
    BlockFull(Aabb<N>),
    OutOfBounds(Aabb<N>),
}

#[derive(Debug, Clone)]
pub struct InsertionError<N: Scalar> {
    error_type: InsertionErrorType<N>,
}

impl<N: Scalar> fmt::Display for InsertionError<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.error_type)
    }
}

impl<N: Scalar> std::error::Error for InsertionError<N> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
