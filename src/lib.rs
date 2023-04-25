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
    slice::Iter,
    sync::{atomic::AtomicUsize, Arc, Weak},
};

use crate::error::{Error, Result};
use aabb::{vector_abs, Aabb, NumTraits};
use nalgebra::{Scalar, Vector3};
use num::{traits::Bounded, NumCast};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
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
        let aabb = self.as_ref().aabb;
        OctreeIter {
            elements: self.query_range(aabb).into_iter(),
            phantom: PhantomData,
        }
    }
}

impl<N, T, const S: usize> FromIterator<T> for Octree<N, T, S>
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

        let mut octree = Octree::new(
            Aabb::from_extents(smallest, largest),
            None,
            container.clone(),
        );

        octree.insert_elements(items).ok();

        octree
    }
}

pub struct OctreeVec<N: Scalar, T: Copy, const S: usize> {
    container: Vec<Octree<N, T, S>>,
}

impl<N, T, const S: usize> OctreeVec<N, T, S>
where
    N: Scalar,
    T: Copy,
{
    #[must_use]
    pub fn new() -> Self {
        Self {
            container: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            container: Vec::with_capacity(capacity),
        }
    }

    pub fn position(&self, other: &Octree<N, T, S>) -> Option<usize> {
        self.container
            .iter()
            .position(|o| Arc::ptr_eq(&o.get_inner(), &other.get_inner()))
    }

    pub fn insert(&mut self, index: usize, octree: Octree<N, T, S>) {
        self.container.insert(index, octree);
    }

    pub fn len(&self) -> usize {
        self.container.len()
    }

    pub fn iter(&self) -> Iter<Octree<N, T, S>> {
        self.container.iter()
    }
}

pub struct ElementArray<N, T: Copy, const S: usize> {
    array: [Option<T>; S],
    phantom: PhantomData<N>,
    length: usize,
}

impl<N, T, const S: usize> Default for ElementArray<N, T, S>
where
    T: Copy,
{
    fn default() -> Self {
        Self {
            array: [None; S],
            phantom: PhantomData,
            length: 0,
        }
    }
}

impl<N, T, const L: usize> Serialize for ElementArray<N, T, L>
where
    T: Serialize + Copy,
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
                for item in array.iter_mut().take(S) {
                    if let Some(element) = seq.next_element()? {
                        *item = element;
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
    N: Scalar,
    T: Copy,
{
    #[must_use]
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

    pub fn is_empty(&self) -> bool
    where
        T: Send + Sync,
    {
        !self.array.par_iter().any(std::option::Option::is_some)
    }

    pub fn insert(&mut self, element: T) -> Result<(), N>
    where
        T: Send + Sync,
    {
        if let Some(empty) = self.array.par_iter_mut().find_any(|e| e.is_none()) {
            empty.replace(element);
            self.length += 1;
            return Ok(());
        };

        Err(Error::InsertionError(InsertionError {
            error_type: InsertionErrorType::Overflow,
        }))
    }

    pub fn remove(&mut self, point: Vector3<N>) -> Option<T>
    where
        N: Send + Sync,
        T: PointData<N> + Send + Sync,
    {
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

    pub fn elements_in_range(&self, range: Aabb<N>) -> Vec<T>
    where
        N: NumTraits + Copy + Send + Sync,
        T: PointData<N> + Send + Sync,
    {
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

    pub fn remove_range(&mut self, range: Aabb<N>)
    where
        N: NumTraits + Copy + Send + Sync,
        T: PointData<N> + Send + Sync,
    {
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
    pub fn drain_duplicates(&mut self, elements: &mut Vec<T>)
    where
        N: Send + Sync,
        T: PointData<N> + Send + Sync,
    {
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
    pub fn replace(&mut self, element: T) -> bool
    where
        N: Send + Sync,
        T: PointData<N> + Send + Sync,
    {
        let pt = element.get_point();
        if let Some(dupe_element) = self
            .array
            .par_iter_mut()
            .filter_map(std::option::Option::as_mut)
            .find_any(|element| element.get_point() == pt)
        {
            *dupe_element = element;
            return true;
        }

        false
    }

    pub fn find_at_point(&self, point: Vector3<N>) -> Option<T>
    where
        N: Send + Sync,
        T: PointData<N> + Send + Sync,
    {
        if let Some(found) = self
            .array
            .par_iter()
            .find_any(|element| element.map(|e| e.get_point() == point).unwrap_or(false))
        {
            return *found;
        }

        None
    }
}

struct OctreeInner<N: Scalar, T: Copy, const S: usize> {
    aabb: Aabb<N>,
    id: usize,
    elements: Arc<RwLock<ElementArray<N, T, S>>>,
    children: [Option<Octree<N, T, S>>; 8],
    parent: Option<Octree<N, T, S>>,
    container: Arc<RwLock<OctreeVec<N, T, S>>>,
    paternity: Paternity,
    lock: Weak<RwLock<Self>>,
}

// TODO: A serialization implementation which serializes just the container, and a deserializer
// which deserializes from that
#[derive(Clone)]
pub struct Octree<N: Scalar, T: Copy, const S: usize> {
    inner: Arc<RwLock<OctreeInner<N, T, S>>>,
}

#[derive(Serialize, Deserialize)]
struct SerializeOctree<N: Scalar, T: Copy, const L: usize> {
    aabb: Aabb<N>,
    id: usize,
    elements: Arc<RwLock<ElementArray<N, T, L>>>,
    children: Vec<usize>,
    parent: Option<usize>,
    paternity: Paternity,
}

impl<N, T, const L: usize> Serialize for Octree<N, T, L>
where
    N: Scalar + Serialize,
    T: Copy + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.as_ref().container.read().len()))?;
        for octree in self.as_ref().container.read().iter() {
            let octree = octree.as_ref();
            seq.serialize_element(&SerializeOctree {
                aabb: octree.aabb.clone(),
                id: octree.id,
                elements: octree.elements.clone(),
                children: octree
                    .children
                    .iter()
                    .flatten()
                    .map(|c| c.as_ref().id)
                    .collect(),
                parent: octree.parent.clone().map(|p| p.as_ref().id),
                paternity: octree.paternity,
            })?
        }
        seq.end()
    }
}

impl<'de, N, T, const S: usize> Deserialize<'de> for Octree<N, T, S>
where
    N: Scalar + Deserialize<'de>,
    T: Copy + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ContainerVisitor<N: Scalar, T: Copy, const S: usize> {
            phantom_t: PhantomData<T>,
            phantom_n: PhantomData<N>,
        }

        impl<'de, N, T, const S: usize> Visitor<'de> for ContainerVisitor<N, T, S>
        where
            N: Scalar + Deserialize<'de>,
            T: Copy + Deserialize<'de>,
        {
            type Value = Octree<N, T, S>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Octree")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut octrees = Vec::new();
                let mut deserialized = Vec::new();
                let container = Arc::new(RwLock::new(OctreeVec::new()));
                while let Ok(Some(octree)) = seq.next_element::<SerializeOctree<N, T, S>>() {
                    let inner = Arc::new(RwLock::new(OctreeInner {
                        id: octree.id,
                        aabb: octree.aabb.clone(),
                        parent: None,
                        paternity: octree.paternity,
                        elements: octree.elements.clone(),
                        container: container.clone(),
                        children: [None, None, None, None, None, None, None, None],
                        lock: Weak::new(),
                    }));

                    let downgrade = Arc::downgrade(&inner);
                    inner.write().lock = downgrade;

                    octrees.push(Octree { inner });

                    deserialized.push(octree);
                }

                container.write().container.extend(octrees.into_iter());

                // Set references
                for (octree, de) in container.read().container.iter().zip(deserialized.iter()) {
                    if let Some(parent_id) = de.parent {
                        octree.get_inner().clone().write().parent = container
                            .read()
                            .container
                            .iter()
                            .find(|o| o.get_inner().read().id == parent_id)
                            .cloned();

                        for child_id in de.children.iter() {
                            let octree = &mut octree.clone();
                            octree
                                .add_child(
                                    container
                                        .read()
                                        .container
                                        .iter()
                                        .find(|o| o.get_inner().read().id == *child_id)
                                        .cloned()
                                        .ok_or_else(|| {
                                            serde::de::Error::custom("Child couldn't be found")
                                        })?,
                                )
                                .map_err(|err| {
                                    serde::de::Error::custom(format!(
                                        "Failed to add child: {}",
                                        err
                                    ))
                                })?
                        }
                    }
                }

                let container_read = container.read();

                Ok(container_read.container[0].clone())
            }
        }

        deserializer.deserialize_seq(ContainerVisitor {
            phantom_t: PhantomData,
            phantom_n: PhantomData,
        })
    }
}

impl<N, T, const S: usize> Octree<N, T, S>
where
    N: Scalar,
    T: Copy,
{
    fn get_inner(&self) -> Arc<RwLock<OctreeInner<N, T, S>>> {
        self.inner.clone()
    }

    fn as_ref(&self) -> RwLockReadGuard<OctreeInner<N, T, S>> {
        self.inner.read()
    }

    fn as_mut(&self) -> RwLockWriteGuard<OctreeInner<N, T, S>> {
        self.inner.write()
    }
}

impl<N, T, const S: usize> Octree<N, T, S>
where
    N: Scalar,
    T: Copy,
{
    pub fn new(
        aabb: Aabb<N>,
        parent: Option<Self>,
        container: Arc<RwLock<OctreeVec<N, T, S>>>,
    ) -> Self
    where
        N: Sync + Send + NumTraits + Copy + Clone,
        T: PointData<N> + PartialEq + Debug + Sync + Send,
    {
        tracing::debug!(target: "creating octree", min = ?aabb.get_min(), max = ?aabb.get_max());

        let octree = Arc::new(RwLock::new(OctreeInner {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            aabb,
            elements: Arc::new(RwLock::new(ElementArray::new())),
            parent,
            children: [None, None, None, None, None, None, None, None],
            container,
            paternity: Paternity::ChildFree,
            lock: Weak::new(),
        }));

        let weak = Arc::downgrade(&octree);
        octree.write().lock = weak;

        let octree_lock = octree.read();
        let container = octree_lock.container.clone();

        let index = octree_lock
            .parent
            .clone()
            .and_then(|p| container.read().position(&p))
            .unwrap_or(0);

        let ret = Octree {
            inner: octree.clone(),
        };
        octree_lock.container.write().insert(index, ret.clone());

        ret
    }

    pub fn get_aabb(&self) -> Aabb<N>
    where
        N: Copy,
    {
        self.as_ref().aabb
    }

    fn paternity(&self) -> Paternity {
        self.as_ref().paternity
    }

    /// # Errors
    /// will return an error if the subdivision does not have the correct dimensions, or if there
    /// are more than 8 children
    #[allow(clippy::too_many_lines)]
    fn subdivide(&mut self) -> Result<(), N>
    where
        N: Sync + Send + NumTraits + Copy + Clone,
        T: PointData<N> + PartialEq + Debug + Sync + Send,
    {
        let zero: N = NumCast::from(0).unwrap();
        let one: N = NumCast::from(1).unwrap();
        let two: N = NumCast::from(2).unwrap();

        // Hacky way of checking if it's an integer and then adjusting min so all values behave like indices
        let adj = if one / two == zero { one } else { zero };

        let min = self.as_ref().aabb.get_min();
        let max = self.as_ref().aabb.get_max();

        let dimensions = vector_abs(self.as_ref().aabb.dimensions);

        let smaller_half = dimensions / two;
        let larger_half = dimensions - smaller_half - Vector3::new(adj, adj, adj);

        let (tx, rx) = crossbeam_channel::unbounded::<Self>();

        let parent = self.clone();
        let container = self.as_ref().container.clone();

        rayon::scope(move |s| {
            //down back left
            let sub_max = min + larger_half;

            let downbackleft = Aabb::<N>::from_extents(min, sub_max);
            tx.send(Octree::new(
                downbackleft,
                Some(parent.clone()),
                container.clone(),
            ))
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
                        Some(parent.clone()),
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
                                Some(parent.clone()),
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
                                        Some(parent.clone()),
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
                        Some(parent.clone()),
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
                                Some(parent.clone()),
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
                    tx3.send(Octree::new(
                        upbackleft,
                        Some(parent.clone()),
                        container.clone(),
                    ))
                    .unwrap();

                    if dimensions.x > one {
                        s.spawn(move |_| {
                            //up back right
                            let sub_min =
                                min + Vector3::new(larger_half.x + adj, larger_half.y + adj, zero);
                            let sub_max = Vector3::new(max.x, max.y, sub_min.z + larger_half.z);
                            let upbackright = Aabb::<N>::from_extents(sub_min, sub_max);
                            tx3.send(Octree::new(
                                upbackright,
                                Some(parent.clone()),
                                container.clone(),
                            ))
                            .unwrap();
                        });
                    }
                });
            }
        });

        for received in rx {
            self.add_child(received)?;
        }

        self.as_mut().paternity = Paternity::ProudParent;

        if cfg!(debug_assertions) {
            let total_volume = self
                .as_ref()
                .children
                .par_iter()
                .filter_map(|child| {
                    child.clone().map(|c| {
                        let w = c.as_ref();
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
    pub fn remove_item(&self, point: Vector3<N>)
    where
        N: Sync + Send + NumTraits + Copy + Clone,
        T: PointData<N> + PartialEq + Debug + Sync + Send,
    {
        if let Paternity::ChildFree = self.as_ref().paternity {
            if self.as_ref().elements.read().is_empty() {
                return;
            }
        }

        if self.as_ref().elements.write().remove(point).is_none() {
            if let Paternity::ProudParent = self.as_ref().paternity {
                self.as_ref().children.par_iter().for_each(|child| {
                    if let Some(child) = child {
                        child.as_ref().elements.write().remove(point);
                    }
                });
            }
        }
    }

    /// Removes all elements which fit inside range, silently avoiding positions that do not fit inside the octree
    pub fn remove_range(&self, range: Aabb<N>)
    where
        N: Sync + Send + NumTraits + Copy + Clone,
        T: PointData<N> + PartialEq + Debug + Sync + Send,
    {
        if let Paternity::ChildFree = self.as_ref().paternity {
            if self.as_ref().elements.read().is_empty() {
                return;
            }
        }

        self.as_ref().elements.write().remove_range(range);

        if let Paternity::ProudParent = self.as_ref().paternity {
            self.as_ref().children.par_iter().for_each(|child| {
                if let Some(child) = child {
                    child.remove_range(range);
                }
            });
        }
    }

    pub fn insert_elements(&mut self, elements: Vec<T>) -> Result<(), N>
    where
        N: Sync + Send + NumTraits + Copy + Clone,
        T: PointData<N> + PartialEq + Debug + Sync + Send,
    {
        let mut elements = elements
            .into_par_iter()
            .filter(|element| self.as_ref().aabb.contains_point(element.get_point()))
            .collect::<Vec<T>>();

        if elements.is_empty() {
            return Err(Error::InsertionError(InsertionError {
                error_type: InsertionErrorType::OutOfBounds(self.as_ref().aabb),
            }));
        }

        let available = S - self.as_ref().elements.read().len();

        self.as_ref()
            .elements
            .write()
            .drain_duplicates(&mut elements);

        let remaining = elements.split_off(available.min(elements.len()));

        let len = self.as_ref().elements.read().len();
        let elements_clone = self.as_ref().elements.clone();
        let paternity = self.as_ref().paternity;
        match self.paternity() {
            Paternity::ChildFree | Paternity::ProudParent if S > len => {
                // TODO: an insert_many method in ElementArray for parallel insertion
                elements
                    .iter()
                    .map(|e| elements_clone.clone().write().insert(*e))
                    .collect::<Result<Vec<_>, _>>()?;

                let len = self.as_ref().elements.read().len();
                if paternity == Paternity::ChildFree && len == S {
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

        let paternity = self.as_ref().paternity;

        match paternity {
            Paternity::ProudParent => self
                .as_mut()
                .children
                .par_iter_mut()
                .filter_map(|child| child.as_mut().map(|c| c.insert_elements(remaining.clone())))
                .filter(std::result::Result::is_ok)
                .collect::<Result<Vec<_>, N>>()
                .map(|_| ()),
            Paternity::ChildFree => Err(Error::<N>::InsertionError(InsertionError {
                error_type: InsertionErrorType::Empty,
            })),
        }
    }

    fn add_child(&mut self, child: Self) -> Result<(), N>
    where
        N: Scalar,
        T: Copy,
    {
        match self
            .as_mut()
            .children
            .iter_mut()
            .find(|child| child.is_none())
        {
            Some(slot) => {
                *slot = Some(child);
                Ok(())
            }
            None => Err(Error::SubdivisionError(SubdivisionError {
                error_type: SubdivisionErrorType::ChildrenFull,
            })),
        }
    }

    pub fn insert(&mut self, element: T) -> Result<(), N>
    where
        N: Sync + Send + NumTraits + Copy + Clone,
        T: PointData<N> + PartialEq + Debug + Sync + Send,
    {
        let pt = element.get_point();

        if !self.as_ref().aabb.contains_point(pt) {
            return Err(Error::<N>::InsertionError(InsertionError {
                error_type: InsertionErrorType::OutOfBounds(self.as_ref().aabb),
            }));
        }

        //if element already exists at point, replace it
        if self.as_ref().elements.write().replace(element) {
            return Ok(());
        }

        let len = self.as_ref().elements.read().len();

        //do first match because you still need to insert into children after subdividing, not either/or
        match &self.paternity() {
            Paternity::ChildFree | Paternity::ProudParent if S > len => {
                self.as_ref().elements.write().insert(element)?;

                return Ok(());
            }

            Paternity::ChildFree => self.subdivide()?,
            Paternity::ProudParent => {}
        }

        match &self.paternity() {
            Paternity::ProudParent => self
                .as_mut()
                .children
                .par_iter_mut()
                .filter_map(|child| child.as_mut().map(|c| c.insert(element)))
                .filter(std::result::Result::is_ok)
                .collect::<Result<Vec<_>, N>>()
                .map(|_| ()),
            Paternity::ChildFree => Err(Error::<N>::InsertionError(InsertionError {
                error_type: InsertionErrorType::Empty,
            })),
        }
    }

    pub fn count(&self) -> usize {
        let mut count: usize = self.as_ref().elements.read().len();

        match &self.as_ref().paternity {
            Paternity::ChildFree => count,
            Paternity::ProudParent => {
                for child in self.as_ref().children.iter().flatten() {
                    count += child.count();
                }
                count
            }
        }
    }

    pub fn query_point(&self, point: Vector3<N>) -> Option<T>
    where
        N: Sync + Send + NumTraits + Copy + Clone,
        T: PointData<N> + PartialEq + Debug + Sync + Send,
    {
        if !self.as_ref().aabb.contains_point(point) {
            return None;
        }

        if let Some(found) = self.as_ref().elements.read().find_at_point(point) {
            return Some(found);
        }

        if let Paternity::ChildFree = self.as_ref().paternity {
            return None;
        }

        self.as_ref()
            .children
            .par_iter()
            .find_map_any(|child| child.as_ref().and_then(|c| c.query_point(point)))
    }

    pub fn query_range(&self, range: Aabb<N>) -> Vec<T>
    where
        N: Sync + Send + NumTraits + Copy + Clone,
        T: PointData<N> + PartialEq + Debug + Sync + Send,
    {
        let volume = range.dimensions.x * range.dimensions.y * range.dimensions.z;
        let mut elements_in_range: Vec<T> = Vec::with_capacity(NumCast::from(volume).unwrap());

        if !self.as_ref().aabb.intersects_bounds(range) {
            return elements_in_range;
        }

        if let Paternity::ChildFree = self.as_ref().paternity {
            if self.as_ref().elements.read().is_empty() {
                return elements_in_range;
            }
        }

        elements_in_range.extend(self.as_ref().elements.read().elements_in_range(range));

        if let Paternity::ChildFree = self.as_ref().paternity {
            return elements_in_range;
        }

        let (tx, rx) = crossbeam_channel::unbounded::<Vec<T>>();

        self.as_ref()
            .children
            .par_iter()
            .for_each_with(tx, |tx, child| {
                if let Some(child) = child {
                    tx.send(child.query_range(range)).unwrap();
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
