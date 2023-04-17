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
    marker::PhantomData,
    sync::{atomic::AtomicUsize, Arc, Weak},
};

use crate::error::{Error, Result};
use aabb::{vector_abs, Aabb, NumTraits};
use nalgebra::{Scalar, Vector3};
use num::{traits::Bounded, NumCast};
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{de::Visitor, ser::SerializeSeq, Deserialize, Deserializer, Serialize};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

pub trait PointData<N: Scalar>: Copy {
    fn get_point(&self) -> Vector3<N>;
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Copy, Clone)]
#[allow(dead_code)]
enum Paternity {
    ProudParent,
    ChildFree,
}

pub struct OctreeIter<N, T> {
    elements: std::vec::IntoIter<T>,
    phantom: PhantomData<N>,
}

impl<N: Scalar, T: PointData<N>> Iterator for OctreeIter<N, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.elements.next()
    }
}

impl<N, T, const S: usize> IntoIterator for Octree<N, T, S>
where
    N: NumTraits + Sync + Send + Copy + Clone,
    T: PointData<N> + PartialEq + Debug + Sync + Send,
{
    type Item = T;
    type IntoIter = OctreeIter<N, T>;
    fn into_iter(self) -> Self::IntoIter {
        let aabb = self.aabb;
        OctreeIter {
            elements: self.query_range(aabb).into_iter(),
            phantom: PhantomData,
        }
    }
}

pub struct OctreeProvider<N: Scalar, T: Copy, const S: usize> {
    octree: Arc<RwLock<Octree<N, T, S>>>,
    container: Arc<RwLock<OctreeVec<N, T, S>>>,
}

impl<N, T, const S: usize> OctreeProvider<N, T, S>
where
    N: Scalar,
    T: Copy,
{
    pub fn get_octree(&self) -> Arc<RwLock<Octree<N, T, S>>> {
        self.octree.clone()
    }

    pub fn get_container(&self) -> Arc<RwLock<OctreeVec<N, T, S>>> {
        self.container.clone()
    }
}

impl<N, T, const S: usize> FromIterator<T> for OctreeProvider<N, T, S>
where
    N: Sync + Send + Bounded + NumTraits + Copy + Clone,
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

        let container = Arc::new(RwLock::new(OctreeVec {
            container: Vec::with_capacity(items.len() / S),
        }));

        let octree = Octree::new(
            Aabb::from_extents(smallest, largest),
            None,
            container.clone(),
        );

        octree.write().insert_elements(items).ok();

        OctreeProvider { octree, container }
    }
}

#[derive(Serialize, Deserialize)]
pub struct OctreeVec<N: Scalar, T: Copy, const S: usize> {
    #[serde(skip)]
    container: Vec<Arc<RwLock<Octree<N, T, S>>>>,
}

impl<N, T, const S: usize> OctreeVec<N, T, S>
where
    N: Scalar,
    T: PointData<N>,
{
    pub fn new() -> Self {
        Self {
            container: Vec::new(),
        }
    }

    pub fn position(&self, other: &Arc<RwLock<Octree<N, T, S>>>) -> Option<usize> {
        self.container.iter().position(|o| Arc::ptr_eq(o, &other))
    }

    pub fn insert(&mut self, index: usize, octree: Arc<RwLock<Octree<N, T, S>>>) {
        self.container.insert(index, octree);
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FakeElementArray<N, T, const S: usize> {
    array: [Option<T>; 5],
    phantom: PhantomData<N>,
    length: usize,
}

pub struct ElementArray<N, T, const S: usize> {
    array: [Option<T>; S],
    phantom: PhantomData<N>,
    length: usize,
}

impl<N, T, const L: usize> Serialize for ElementArray<N, T, L>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(L))?;
        for element in &self.array {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

impl<'de, N, T, const S: usize> Deserialize<'de> for ElementArray<N, T, S>
where
    T: Copy + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ElementArrayVisitor<N, T, const S: usize> {
            phantom_n: PhantomData<N>,
            phantom_t: PhantomData<T>,
        }

        impl<'de, N, T, const S: usize> Visitor<'de> for ElementArrayVisitor<N, T, S>
        where
            T: Copy + Deserialize<'de>,
        {
            type Value = ElementArray<N, T, S>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ElementArray")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut array = [None; S];
                let mut length = 0;
                for i in 0..S {
                    if let Some(element) = seq.next_element()? {
                        array[i] = element;
                        length += 1;
                    }
                }

                Ok(ElementArray {
                    array,
                    phantom: PhantomData,
                    length,
                })
            }
        }

        deserializer.deserialize_seq(ElementArrayVisitor {
            phantom_n: PhantomData,
            phantom_t: PhantomData,
        })
    }
}

impl<N, T, const S: usize> ElementArray<N, T, S>
where
    N: NumTraits + Send + Sync + Copy,
    T: PointData<N> + Send + Sync + Debug,
{
    pub fn new() -> Self {
        Self {
            array: [None; S],
            phantom: PhantomData,
            length: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        !self.array.par_iter().any(|e| e.is_some())
    }

    pub fn insert(&mut self, element: T) -> Result<(), N> {
        if let Some(empty) = self.array.par_iter_mut().find_any(|e| e.is_none()) {
            *empty = Some(element);
            self.length += 1;
            return Ok(());
        };

        Err(Error::InsertionError(InsertionError {
            error_type: InsertionErrorType::Overflow,
        }))
    }

    pub fn remove(&mut self, point: Vector3<N>) -> Option<T> {
        if let Some(element) = self
            .array
            .par_iter_mut()
            .find_any(|element| element.map(|e| e.get_point() == point).unwrap_or(false))
        {
            self.length -= 1;
            return element.take();
        }
        None
    }

    pub fn elements_in_range(&self, range: Aabb<N>) -> Vec<T> {
        self.array
            .par_iter()
            .filter(|element| {
                element
                    .map(|e| range.contains_point(e.get_point()))
                    .unwrap_or(false)
            })
            .map(|e| e.unwrap())
            .collect::<Vec<T>>()
    }

    pub fn remove_range(&mut self, range: Aabb<N>) {
        self.length -= self
            .array
            .par_iter_mut()
            .filter(|element| {
                element
                    .map(|e| range.contains_point(e.get_point()))
                    .unwrap_or(false)
            })
            .map(|element| {
                _ = element.take();
            })
            .count();
    }

    /// Drains duplicates from the Vec<T>, replacing existing elements at their respective points
    pub fn drain_duplicates(&mut self, elements: &mut Vec<T>) {
        //overwrite duplicates
        self.array.par_iter_mut().for_each(|element| {
            if let Some(dupe) = elements.iter().find(|inc| {
                element
                    .map(|e| e.get_point() == inc.get_point())
                    .unwrap_or(false)
            }) {
                element.replace(*dupe);
            }
        });

        //cull out duplicates
        elements.retain(|inc| {
            !self.array.par_iter().any(|orig| {
                orig.map(|e| e.get_point() == inc.get_point())
                    .unwrap_or(false)
            })
        });
    }

    /// Replaces an element at its position if possible, returning true if so.
    pub fn replace(&mut self, element: T) -> bool {
        let pt = element.get_point();
        if let Some(dupe_element) = self
            .array
            .par_iter_mut()
            .filter_map(|element| element.as_mut())
            .find_any(|element| element.get_point() == pt)
        {
            *dupe_element = element;
            return true;
        }

        false
    }

    pub fn find_at_point(&self, point: Vector3<N>) -> Option<T> {
        if let Some(found) = self
            .array
            .par_iter()
            .find_any(|element| element.map(|e| e.get_point() == point).unwrap_or(false))
        {
            return found.to_owned();
        }

        None
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Octree<N: Scalar, T: Copy, const S: usize> {
    aabb: Aabb<N>,
    id: usize,
    elements: Arc<RwLock<ElementArray<N, T, S>>>,
    children: [Option<Arc<RwLock<Self>>>; 8],
    parent: Option<Arc<RwLock<Self>>>,
    container: Arc<RwLock<OctreeVec<N, T, S>>>,
    paternity: Paternity,
    #[serde(skip)]
    lock: Arc<Weak<RwLock<Self>>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FakeOctree<N: Scalar, T, const S: usize> {
    aabb: Aabb<N>,
    id: usize,
    elements: Arc<RwLock<FakeElementArray<N, T, S>>>,
    phantom_n: PhantomData<N>,
    phantom_t: PhantomData<T>,
}

impl<N, T, const S: usize> Octree<N, T, S>
where
    N: Sync + Send + NumTraits + Copy + Clone,
    T: PointData<N> + PartialEq + Debug + Sync + Send,
{
    pub fn new(
        aabb: Aabb<N>,
        parent: Option<Arc<RwLock<Self>>>,
        container: Arc<RwLock<OctreeVec<N, T, S>>>,
    ) -> Arc<RwLock<Self>> {
        tracing::debug!(target: "creating octree", min = ?aabb.get_min(), max = ?aabb.get_max());

        let octree = Arc::new(RwLock::new(Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            aabb,
            elements: Arc::new(RwLock::new(ElementArray::new())),
            parent,
            children: [None, None, None, None, None, None, None, None],
            container,
            paternity: Paternity::ChildFree,
            lock: Arc::new(Weak::new()),
        }));

        let weak = Arc::downgrade(&octree);
        octree.write().lock = Arc::new(weak.clone());

        let octree_lock = octree.read();
        let container = octree_lock.container.clone();

        let index = octree_lock
            .parent
            .clone()
            .and_then(|p| container.read().position(&p))
            .unwrap_or(0);
        octree_lock.container.write().insert(index, octree.clone());

        octree.clone()
    }

    pub fn get_aabb(&self) -> Aabb<N> {
        self.aabb
    }

    /// # Errors
    /// will return an error if the subdivision does not have the correct dimensions, or if there
    /// are more than 8 children
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

        let (tx, rx) = crossbeam_channel::unbounded::<Arc<RwLock<Self>>>();

        let parent = self.lock.upgrade().clone();
        let container = self.container.clone();

        rayon::scope(move |s| {
            //down back left
            let sub_max = min + larger_half;

            let downbackleft = Aabb::<N>::from_extents(min, sub_max);
            tx.send(Octree::new(downbackleft, parent.clone(), container.clone()))
                .unwrap();

            if dimensions.x > one {
                let tx1 = tx.clone();

                let container = container.clone();
                let parent = parent.clone();

                s.spawn(move |s| {
                    let sub_min = min + Vector3::new(larger_half.x + adj, zero, zero);
                    let sub_max =
                        Vector3::new(max.x, sub_min.y + larger_half.y, sub_min.z + larger_half.z);

                    let downbackright = Aabb::<N>::from_extents(sub_min, sub_max);
                    tx1.send(Octree::new(
                        downbackright,
                        parent.clone(),
                        container.clone(),
                    ))
                    .unwrap();

                    if dimensions.z > one {
                        s.spawn(move |s| {
                            //down forward right
                            let sub_min =
                                min + Vector3::new(larger_half.x + adj, zero, larger_half.z + adj);
                            let sub_max = Vector3::new(max.x, sub_min.y + larger_half.y, max.z);
                            let downforwardright = Aabb::<N>::from_extents(sub_min, sub_max);
                            tx1.send(Octree::new(
                                downforwardright,
                                parent.clone(),
                                container.clone(),
                            ))
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
                                    tx1.send(Octree::new(
                                        upforwardright,
                                        parent.clone(),
                                        container.clone(),
                                    ))
                                    .unwrap();
                                });
                            }
                        });
                    }
                });
            }

            if dimensions.z > one {
                let tx2 = tx.clone();

                let container = container.clone();
                let parent = parent.clone();

                s.spawn(move |s| {
                    //down forward left
                    let sub_min = min + Vector3::new(zero, zero, larger_half.z + adj);
                    let sub_max =
                        Vector3::new(sub_min.x + larger_half.x, sub_min.y + larger_half.y, max.z);

                    let downforwardleft = Aabb::<N>::from_extents(sub_min, sub_max);
                    tx2.send(Octree::new(
                        downforwardleft,
                        parent.clone(),
                        container.clone(),
                    ))
                    .unwrap();

                    if dimensions.y > one {
                        s.spawn(move |_| {
                            //up forward left
                            let sub_min =
                                min + Vector3::new(zero, larger_half.y + adj, larger_half.z + adj);
                            let sub_max = Vector3::new(sub_min.x + larger_half.x, max.y, max.z);
                            let upforwardleft = Aabb::<N>::from_extents(sub_min, sub_max);
                            tx2.send(Octree::new(
                                upforwardleft,
                                parent.clone(),
                                container.clone(),
                            ))
                            .unwrap();
                        });
                    }
                });
            }

            if dimensions.y > one {
                let tx3 = tx.clone();

                s.spawn(move |s| {
                    //up back left
                    let sub_min = min + Vector3::new(zero, larger_half.y + adj, zero);
                    let sub_max =
                        Vector3::new(sub_min.x + larger_half.x, max.y, sub_min.z + larger_half.z);
                    let upbackleft = Aabb::<N>::from_extents(sub_min, sub_max);
                    tx3.send(Octree::new(upbackleft, parent.clone(), container.clone()))
                        .unwrap();

                    if dimensions.x > one {
                        s.spawn(move |_| {
                            //up back right
                            let sub_min =
                                min + Vector3::new(larger_half.x + adj, larger_half.y + adj, zero);
                            let sub_max = Vector3::new(max.x, max.y, sub_min.z + larger_half.z);
                            let upbackright = Aabb::<N>::from_extents(sub_min, sub_max);
                            tx3.send(Octree::new(upbackright, parent.clone(), container.clone()))
                                .unwrap();
                        });
                    }
                });
            }
        });

        for received in rx {
            self.add_child(received)?;
        }

        self.paternity = Paternity::ProudParent;

        if cfg!(debug_assertions) {
            let total_volume = self
                .children
                .par_iter()
                .filter_map(|child| {
                    child.clone().map(|c| {
                        let w = c.read();
                        w.aabb.dimensions.x * w.aabb.dimensions.y * w.aabb.dimensions.z
                    })
                })
                .sum();

            let volume = dimensions.x * dimensions.y * dimensions.z;

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
            if self.elements.read().is_empty() {
                return;
            }
        }

        if let None = self.elements.write().remove(point) {
            if let Paternity::ProudParent = self.paternity {
                self.children.par_iter_mut().for_each(|child| {
                    if let Some(child) = child {
                        child.read().elements.write().remove(point);
                    }
                });
            }
        }
    }

    /// Removes all elements which fit inside range, silently avoiding positions that do not fit inside the octree
    pub fn remove_range(&mut self, range: Aabb<N>) {
        if let Paternity::ChildFree = self.paternity {
            if self.elements.read().is_empty() {
                return;
            }
        }

        self.elements.write().remove_range(range);

        if let Paternity::ProudParent = self.paternity {
            self.children.par_iter_mut().for_each(|child| {
                if let Some(child) = child {
                    child.write().remove_range(range);
                }
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

        let available = S - self.elements.read().len();

        self.elements.write().drain_duplicates(&mut elements);

        let remaining = elements.split_off(available.min(elements.len()));

        let len = self.elements.read().len();
        let elements_clone = self.elements.clone();
        match &self.paternity {
            Paternity::ChildFree | Paternity::ProudParent if S > len => {
                // TODO: an insert_many method in ElementArray for parallel insertion
                elements
                    .iter()
                    .map(|e| elements_clone.clone().write().insert(*e))
                    .collect::<Result<Vec<_>, _>>()?;

                let len = self.elements.read().len();
                if self.paternity == Paternity::ChildFree && len == S {
                    self.subdivide()?;
                }

                if len > S {
                    return Err(Error::InsertionError(InsertionError {
                        error_type: InsertionErrorType::Overflow,
                    }));
                }
            }
            Paternity::ChildFree => self.subdivide()?,
            Paternity::ProudParent => {}
        }

        if remaining.is_empty() {
            return Ok(());
        }

        match &self.paternity {
            Paternity::ProudParent => self
                .children
                .par_iter_mut()
                .filter_map(|child| {
                    child
                        .as_mut()
                        .map(|c| c.write().insert_elements(remaining.clone()))
                })
                .filter(|r| r.is_ok())
                .collect::<Result<Vec<_>, N>>()
                .map(|_| ()),
            Paternity::ChildFree => Err(Error::<N>::InsertionError(InsertionError {
                error_type: InsertionErrorType::Empty,
            })),
        }
    }

    fn add_child(&mut self, child: Arc<RwLock<Self>>) -> Result<(), N> {
        match self.children.iter_mut().find(|child| child.is_none()) {
            Some(slot) => {
                *slot = Some(child);
                Ok(())
            }
            None => Err(Error::SubdivisionError(SubdivisionError {
                error_type: SubdivisionErrorType::ChildrenFull,
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
        if self.elements.write().replace(element) {
            return Ok(());
        }

        let len = self.elements.read().len();

        //do first match because you still need to insert into children after subdividing, not either/or
        match &self.paternity {
            Paternity::ChildFree | Paternity::ProudParent if S > len => {
                self.elements.write().insert(element)?;

                return Ok(());
            }

            Paternity::ChildFree => self.subdivide()?,
            Paternity::ProudParent => {}
        }

        match &self.paternity {
            Paternity::ProudParent => self
                .children
                .par_iter_mut()
                .filter_map(|child| child.as_mut().map(|c| c.write().insert(element)))
                .filter(|r| r.is_ok())
                .collect::<Result<Vec<_>, N>>()
                .map(|_| ()),
            Paternity::ChildFree => Err(Error::<N>::InsertionError(InsertionError {
                error_type: InsertionErrorType::Empty,
            })),
        }
    }

    pub fn count(&self) -> usize {
        let mut count: usize = self.elements.read().len();

        match &self.paternity {
            Paternity::ChildFree => count,
            Paternity::ProudParent => {
                for child in &self.children {
                    if let Some(child) = child {
                        count += child.read().count();
                    }
                }
                count
            }
        }
    }

    pub fn query_point(&self, point: Vector3<N>) -> Option<T> {
        if !self.aabb.contains_point(point) {
            return None;
        }

        if let Some(found) = self.elements.read().find_at_point(point) {
            return Some(found);
        }

        if let Paternity::ChildFree = self.paternity {
            return None;
        }

        self.children
            .par_iter()
            .find_map_any(|child| child.as_ref().and_then(|c| c.read().query_point(point)))
    }

    pub fn query_range(&self, range: Aabb<N>) -> Vec<T> {
        let volume = range.dimensions.x * range.dimensions.y * range.dimensions.z;
        let mut elements_in_range: Vec<T> = Vec::with_capacity(NumCast::from(volume).unwrap());

        if !self.aabb.intersects_bounds(range) {
            return elements_in_range;
        }

        if let Paternity::ChildFree = self.paternity {
            if self.elements.read().is_empty() {
                return elements_in_range;
            }
        }

        elements_in_range.extend(self.elements.read().elements_in_range(range));

        if let Paternity::ChildFree = self.paternity {
            return elements_in_range;
        }

        let (tx, rx) = crossbeam_channel::unbounded::<Vec<T>>();

        self.children.par_iter().for_each_with(tx, |tx, child| {
            if let Some(child) = child {
                tx.send(child.read().query_range(range)).unwrap();
            }
        });

        for mut received in rx {
            elements_in_range.append(&mut received);
        }

        elements_in_range
    }
}

#[derive(Clone, Debug)]
pub enum SubdivisionErrorType<N: Scalar> {
    IncorrectDimensions(N, N),
    ChildrenFull,
    MissingFromContainer,
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
    DroppedSelf,
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
