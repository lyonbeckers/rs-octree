#![allow(clippy::cast_sign_loss)]

use std::collections::HashSet;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use tracing::Subscriber;
use tracing_subscriber::EnvFilter;

use crate::{error, geometry::aabb, Octree, PointData, DEFAULT_MAX};

type Point = Vector3<i32>;
type Aabb = aabb::Aabb<i32>;

type FloatPoint = Vector3<f32>;
type FloatAabb = aabb::Aabb<f32>;

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
pub struct FloatTileData {
    point: FloatPoint,
}

impl FloatTileData {
    pub fn new(point: FloatPoint) -> Self {
        FloatTileData { point }
    }
}

impl PointData<f32> for FloatTileData {
    fn get_point(&self) -> FloatPoint {
        self.point
    }
}

#[derive(Serialize, Deserialize, PartialEq, Hash, Eq, Copy, Clone, Debug)]
pub struct TileData {
    point: Point,
}

impl TileData {
    pub fn new(point: Point) -> Self {
        TileData { point }
    }
}

impl PointData<i32> for TileData {
    fn get_point(&self) -> Point {
        self.point
    }
}

fn setup_subscriber() -> impl Subscriber {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .finish()
}

#[test]
fn from_iter() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let pts: Vec<TileData> = vec![TileData::new(Point::zeros())];

    let mut oct_a: Octree<i32, TileData> = Octree::new(
        Aabb::from_extents(Point::zeros(), Point::zeros()),
        DEFAULT_MAX,
    );

    for pt in &pts {
        oct_a.insert(*pt).unwrap();
    }

    let oct_b = pts.into_iter().collect::<Octree<i32, TileData>>();

    assert_eq!(
        oct_a.into_iter().collect::<Vec<TileData>>(),
        oct_b.into_iter().collect::<Vec<TileData>>()
    );
}

#[test]
fn test_float() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = FloatAabb::new(FloatPoint::new(1., 1., 1.), FloatPoint::new(4., 4., 4.));
    let mut octree = Octree::<f32, FloatTileData>::new(aabb, DEFAULT_MAX);

    octree
        .insert(FloatTileData::new(FloatPoint::new(
            1.2233,
            1.666_778,
            1.999_888_8,
        )))
        .ok();

    assert!(octree
        .query_point(FloatPoint::new(1.2233, 1.666_778, 1.999_888_8))
        .is_some());

    octree
        .insert(FloatTileData::new(FloatPoint::new(
            1.2233,
            1.666_778,
            1.999_888_8,
        )))
        .ok();

    assert_eq!(octree.into_iter().count(), 1);
}

#[test]
fn even_subdivision() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::new(Point::new(1, 1, 1), Point::new(4, 4, 4));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(octree.count(), count);
}

#[test]
fn odd_subdivision() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::new(Point::new(1, 1, 1), Point::new(5, 5, 5));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(octree.count(), count);
}

#[test]
fn tiny_test() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::from_extents(Point::new(2, -1, 2), Point::new(3, 0, 3));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(octree.count(), count);
}

#[test]
fn large_test() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::from_extents(Point::new(0, 0, 0), Point::new(9, 9, 9));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(octree.count(), count);
}

#[test]
fn large_test_small_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::from_extents(Point::new(0, 0, 0), Point::new(9, 9, 9));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(octree.count(), count);
}

#[test]
fn contains_point() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::new(Point::new(-5, 5, 5), Point::new(-10, 10, 10));

    assert!(!aabb.contains_point(Point::zeros()));
}

#[test]
fn from_extents() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let min = Point::new(0, 0, 0);
    let max = Point::new(9, 9, 9);

    let aabb = Aabb::from_extents(min, max);
    let other = Aabb::new(Point::new(5, 5, 5), Point::new(10, 10, 10));
    assert_eq!(aabb.get_min(), other.get_min());
    assert_eq!(aabb.get_max(), other.get_max());
    assert_eq!(aabb.center, other.center);
}

#[test]
fn volume_one() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::new(Point::zeros(), Point::new(1, 1, 1));

    println!("min {:?} max {:?}", aabb.get_min(), aabb.get_max());
    assert!(aabb.contains_point(Point::zeros()));
}

#[test]
fn overwrite_element() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::new(Point::new(0, 0, 0), Point::new(4, 4, 4));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert!(octree.insert(TileData::new(Point::zeros())).is_ok());
    assert_eq!(octree.into_iter().count(), count);
}

#[test]
fn overwrite_all() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::new(Point::new(0, 0, 0), Point::new(9, 9, 9));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();
    fill_octree(aabb, &mut octree, &mut count).unwrap();
}

#[test]
fn query_point() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::new(Point::new(0, 0, 0), Point::new(4, 4, 4));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert!(octree.get_aabb().contains_point(Point::new(0, 1, 0)));
    assert!(octree.query_point(Point::new(0, 1, 0)).is_some());
    assert!(octree.query_point(Point::new(0, 3, 0)).is_none());
}

#[test]
fn remove_range_tiny_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::from_extents(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    fill_octree(aabb, &mut octree, &mut 0).unwrap();

    let before = octree.clone().into_iter().count();

    octree.remove_range(Aabb::from_extents(Point::new(0, 0, 0), Point::new(0, 0, 0)));

    assert_eq!(octree.into_iter().count(), before - 1);
}

#[test]
fn query_range_tiny_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();
    let aabb = Aabb::from_extents(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();
    assert_eq!(
        octree
            .query_range(Aabb::from_extents(Point::zeros(), Point::zeros()))
            .len(),
        1
    );
    assert_eq!(octree.into_iter().count(), count);
}

#[test]
fn iter_count_tiny_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::from_extents(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(count, octree.into_iter().count());
}

#[test]
fn contains_point_tiny_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::from_extents(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    fill_octree(aabb, &mut octree, &mut 0).unwrap();

    assert!(aabb.contains_point(Point::new(0, 0, 0)));
    assert!(aabb.contains_point(Point::new(1, 0, 0)));
    assert!(aabb.contains_point(Point::new(2, 0, 0)));
}

#[test]
fn remove_element() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::new(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    fill_octree(aabb, &mut octree, &mut 0).unwrap();

    let range = Aabb::from_extents(Point::new(0, 0, 0), Point::new(0, 0, 0));

    assert!(octree.query_point(Point::new(0, 0, 0)).is_some());
    octree.remove_range(range);
    assert!(octree.query_point(Point::new(0, 0, 0)).is_none());
}

#[test]
fn remove_all() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = Aabb::from_extents(Point::new(0, 0, 0), Point::new(9, 9, 9));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count).unwrap();
    assert_eq!(
        octree.clone().into_iter().count(),
        (aabb.dimensions.x * aabb.dimensions.y * aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    println!("filling...");
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(
        octree.clone().into_iter().count(),
        (aabb.dimensions.x * aabb.dimensions.y * aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    println!("filling...");
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(
        octree.clone().into_iter().count(),
        (aabb.dimensions.x * aabb.dimensions.y * aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    println!("filling...");
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(
        octree.clone().into_iter().count(),
        (aabb.dimensions.x * aabb.dimensions.y * aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    println!("filling...");
    fill_octree(aabb, &mut octree, &mut count).unwrap();

    assert_eq!(
        octree.clone().into_iter().count(),
        (aabb.dimensions.x * aabb.dimensions.y * aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    assert!(octree.into_iter().count() == 0)
}

#[test]
fn serialize_deserialize() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let mut octree = Octree::<i32, TileData>::new(
        Aabb::from_extents(Point::new(-5, -5, -5), Point::new(5, 5, 5)),
        DEFAULT_MAX,
    );

    octree.insert(TileData::new(Point::new(1, 0, 0))).unwrap();
    octree.insert(TileData::new(Point::new(0, 1, 0))).unwrap();
    octree.insert(TileData::new(Point::new(0, 0, 1))).unwrap();
    octree.insert(TileData::new(Point::new(-1, 0, 0))).unwrap();
    octree.insert(TileData::new(Point::new(0, -1, 0))).unwrap();
    octree.insert(TileData::new(Point::new(0, 0, -1))).unwrap();

    let octree_clone = octree.clone();

    let pretty = ron::ser::PrettyConfig::default();
    let ser_ron = match ron::ser::to_string_pretty(&octree, pretty) {
        Ok(r) => {
            println!("{:?}", r);
            r
        }
        Err(err) => {
            panic!("{:?}", err);
        }
    };

    let round_trip: Octree<i32, TileData> = ron::de::from_str(&ser_ron).unwrap();

    assert!(
        octree_clone
            .into_iter()
            .collect::<HashSet<TileData>>()
            .symmetric_difference(&round_trip.into_iter().collect::<HashSet<TileData>>())
            .count()
            == 0
    );
}

fn fill_octree(
    aabb: Aabb,
    octree: &mut Octree<i32, TileData>,
    count: &mut usize,
) -> error::Result<(), i32> {
    let min = aabb.get_min();
    let max = aabb.get_max();

    let mut values = Vec::new();

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                *count += 1;

                values.push(TileData::new(Point::new(x, y, z)));
            }
        }
    }

    octree.insert_elements(values)
}

#[test]
fn test_aabb_intersection() {
    let aabb1 = Aabb::from_extents(Point::new(0, 0, 0), Point::new(3, 3, 3));
    let aabb2 = Aabb::from_extents(Point::new(-1, -1, -1), Point::new(2, 2, 2));

    assert_eq!(
        Aabb::from_extents(Point::new(0, 0, 0), Point::new(2, 2, 2)),
        aabb1.get_intersection(aabb2)
    );
}
