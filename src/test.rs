use nalgebra::Vector3;
use ron;
use serde::{Deserialize, Serialize};
use tracing::Subscriber;

use crate::{geometry::aabb, Octree, PointData, DEFAULT_MAX};

type Point = Vector3<i32>;
type AABB = aabb::AABB<i32>;

#[derive(Serialize, Deserialize, Eq, Hash, PartialEq, Clone, Debug)]
pub struct TileData {
    point: Point,
}

impl Copy for TileData {}

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
    tracing_subscriber::fmt().finish()
}

#[test]
fn from_iter() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let pts: Vec<TileData> = vec![TileData::new(Point::zeros())];

    let mut oct_a: Octree<i32, TileData> = Octree::new(
        AABB::from_extents(Point::zeros(), Point::zeros()),
        DEFAULT_MAX,
    );

    for pt in &pts {
        oct_a.insert(*pt).unwrap();
    }

    let oct_b = pts.into_iter().collect::<Octree<i32, TileData>>();

    assert_eq!(oct_a, oct_b);
}

#[test]
fn even_subdivision() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::new(Point::new(1, 1, 1), Point::new(4, 4, 4));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(octree.count(), count);
}

#[test]
fn odd_subdivision() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::new(Point::new(1, 1, 1), Point::new(5, 5, 5));

    println!("{:?} {:?}", aabb.get_min(), aabb.get_max());

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(octree.count(), count);
}

#[test]
fn tiny_test() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::from_extents(Point::new(2, -1, 2), Point::new(3, 0, 3));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(octree.count(), count);
}

#[test]
fn large_test() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::from_extents(Point::new(0, 0, 0), Point::new(9, 9, 9));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(octree.count(), count);
}

#[test]
fn large_test_small_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::from_extents(Point::new(0, 0, 0), Point::new(9, 9, 9));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(octree.count(), count);
}

#[test]
fn contains_point() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::new(Point::new(-5, 5, 5), Point::new(-10, 10, 10));

    println!("min {:?} max {:?}", aabb.get_min(), aabb.get_max());
    assert!(!aabb.contains_point(Point::zeros()));
}

#[test]
fn from_extents() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let min = Point::new(0, 0, 0);
    let max = Point::new(9, 9, 9);

    println!("min {:?} max {:?}", min, max);

    let aabb = AABB::from_extents(min, max);
    let other = AABB::new(Point::new(5, 5, 5), Point::new(10, 10, 10));
    assert_eq!(aabb.get_min(), other.get_min());
    assert_eq!(aabb.get_max(), other.get_max());
    assert_eq!(aabb.center, other.center);
}

#[test]
fn volume_one() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::new(Point::zeros(), Point::new(1, 1, 1));

    println!("min {:?} max {:?}", aabb.get_min(), aabb.get_max());
    assert!(aabb.contains_point(Point::zeros()));
}

#[test]
fn overwrite_element() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::new(Point::new(0, 0, 0), Point::new(4, 4, 4));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);

    assert!(octree.insert(TileData::new(Point::zeros())).is_ok());
    assert_eq!(octree.into_iter().count(), count);
}

#[test]
fn overwrite_all() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::new(Point::new(0, 0, 0), Point::new(9, 9, 9));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);
    fill_octree(aabb, &mut octree, &mut count);
}

#[test]
fn query_point() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::new(Point::new(0, 0, 0), Point::new(4, 4, 4));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);

    assert!(octree.get_aabb().contains_point(Point::new(0, 1, 0)));
    assert!(octree.query_point(Point::new(0, 1, 0)).is_some());
    assert!(octree.query_point(Point::new(0, 3, 0)).is_none());
}

#[test]
fn remove_range_tiny_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::from_extents(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    fill_octree(aabb, &mut octree, &mut 0);

    let before = octree.clone().into_iter().count();

    octree.remove_range(AABB::from_extents(Point::new(0, 0, 0), Point::new(0, 0, 0)));

    assert_eq!(octree.clone().into_iter().count(), before - 1);
}

#[test]
fn query_range_tiny_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();
    let aabb = AABB::from_extents(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);
    assert_eq!(
        octree
            .query_range(AABB::from_extents(Point::zeros(), Point::zeros()))
            .len(),
        1
    );
    assert_eq!(octree.into_iter().count(), count);
}

#[test]
fn iter_count_tiny_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::from_extents(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(count, octree.into_iter().count());
}

#[test]
fn contains_point_tiny_max() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::from_extents(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, 1);

    fill_octree(aabb, &mut octree, &mut 0);

    assert!(aabb.contains_point(Point::new(0, 0, 0)));
    assert!(aabb.contains_point(Point::new(1, 0, 0)));
    assert!(aabb.contains_point(Point::new(2, 0, 0)));
}

#[test]
fn remove_element() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::new(Point::new(0, 0, 0), Point::new(7, 7, 7));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    fill_octree(aabb, &mut octree, &mut 0);

    let range = AABB::from_extents(Point::new(0, 0, 0), Point::new(0, 0, 0));

    assert!(octree.query_point(Point::new(0, 0, 0)).is_some());
    octree.remove_range(range);
    assert!(octree.query_point(Point::new(0, 0, 0)).is_none());
}

#[test]
fn remove_all() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let aabb = AABB::from_extents(Point::new(0, 0, 0), Point::new(9, 9, 9));

    let mut octree = Octree::<i32, TileData>::new(aabb, DEFAULT_MAX);

    let mut count = 0;
    fill_octree(aabb, &mut octree, &mut count);
    assert_eq!(
        octree.clone().into_iter().count(),
        (&aabb.dimensions.x * &aabb.dimensions.y * &aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    println!("filling...");
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(
        octree.clone().into_iter().count(),
        (&aabb.dimensions.x * &aabb.dimensions.y * &aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    println!("filling...");
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(
        octree.clone().into_iter().count(),
        (&aabb.dimensions.x * &aabb.dimensions.y * &aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    println!("filling...");
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(
        octree.clone().into_iter().count(),
        (&aabb.dimensions.x * &aabb.dimensions.y * &aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    println!("filling...");
    fill_octree(aabb, &mut octree, &mut count);

    assert_eq!(
        octree.clone().into_iter().count(),
        (&aabb.dimensions.x * &aabb.dimensions.y * &aabb.dimensions.z) as usize
    );

    println!("removing...");
    octree.remove_range(aabb);

    assert!(octree.into_iter().count() == 0)
}

#[test]
fn serialize_deserialize() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let mut octree = Octree::<i32, TileData>::new(
        AABB::from_extents(Point::new(-5, -5, -5), Point::new(5, 5, 5)),
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

    assert_eq!(octree_clone, round_trip);
}

fn fill_octree(aabb: AABB, octree: &mut Octree<i32, TileData>, count: &mut usize) {
    let min = aabb.get_min();
    let max = aabb.get_max();

    for z in min.z..max.z + 1 {
        for y in min.y..max.y + 1 {
            for x in min.x..max.x + 1 {
                *count += 1;
                match octree.insert(TileData::new(Point::new(x, y, z))) {
                    Ok(_) => {}
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
        }
    }
}

#[test]
fn test_aabb_intersection() {
    let aabb1 = AABB::from_extents(Point::new(0, 0, 0), Point::new(3, 3, 3));
    let aabb2 = AABB::from_extents(Point::new(-1, -1, -1), Point::new(2, 2, 2));

    assert_eq!(
        AABB::from_extents(Point::new(0, 0, 0), Point::new(2, 2, 2)),
        aabb1.get_intersection(aabb2)
    );
}