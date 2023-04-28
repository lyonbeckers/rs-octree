#![feature(test)]
extern crate test;

use std::sync::Arc;

use aabb::Aabb;
use nalgebra::Vector3;
use parking_lot::RwLock;
use test::Bencher;

use octree::{Octree, OctreeVec, PointData};

#[derive(Debug, PartialEq, Copy, Clone)]
struct TileData {
    point: Vector3<i32>,
}

impl PointData<i32> for TileData {
    fn get_point(&self) -> Vector3<i32> {
        self.point
    }
}

#[bench]
fn bench_fill_20e3_octree_32(b: &mut Bencher) {
    let aabb = Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(20, 20, 20));
    let mut points = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            for z in 0..20 {
                points.push(TileData {
                    point: Vector3::new(x, y, z),
                });
            }
        }
    }
    b.iter(|| {
        let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 32>::new()));
        let octree = Octree::<i32, TileData, 32>::new(aabb, None, container);

        octree.insert_elements(&points.clone()).ok();
    });
}

#[bench]
fn bench_fill_20e3_octree_32_with_capacity(b: &mut Bencher) {
    let aabb = Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(20, 20, 20));
    let mut points = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            for z in 0..20 {
                points.push(TileData {
                    point: Vector3::new(x, y, z),
                });
            }
        }
    }

    b.iter(|| {
        // 250 is the theoretical most we'd need
        let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 32>::with_capacity(
            250,
        )));
        let octree = Octree::<i32, TileData, 32>::new(aabb, None, container);

        octree.insert_elements(&points.clone()).ok();
    });
}

#[bench]
fn bench_add_remove_20e3_32(b: &mut Bencher) {
    let aabb = Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(20, 20, 20));
    let mut points = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            for z in 0..20 {
                points.push(TileData {
                    point: Vector3::new(x, y, z),
                });
            }
        }
    }
    let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 32>::with_capacity(
        250,
    )));
    let octree = Octree::<i32, TileData, 32>::new(aabb, None, container);
    octree.insert_elements(&points.clone()).ok();

    b.iter(|| {
        octree.remove_range(aabb);
        octree.insert_elements(&points.clone()).ok()
    });
}

#[bench]
fn bench_add_remove_20e3_128(b: &mut Bencher) {
    let aabb = Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(20, 20, 20));
    let mut points = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            for z in 0..20 {
                points.push(TileData {
                    point: Vector3::new(x, y, z),
                });
            }
        }
    }
    let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 128>::with_capacity(
        250,
    )));
    let octree = Octree::<i32, TileData, 128>::new(aabb, None, container);
    octree.insert_elements(&points.clone()).ok();

    b.iter(|| {
        octree.remove_range(aabb);
        octree.insert_elements(&points.clone()).ok()
    });
}

#[bench]
fn bench_overwrite_20e3_32(b: &mut Bencher) {
    let aabb = Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(20, 20, 20));
    let mut points = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            for z in 0..20 {
                points.push(TileData {
                    point: Vector3::new(x, y, z),
                });
            }
        }
    }
    let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 32>::with_capacity(
        250,
    )));
    let octree = Octree::<i32, TileData, 32>::new(aabb, None, container);
    octree.insert_elements(&points.clone()).ok();

    b.iter(|| octree.insert_elements(&points.clone()).ok());
}

#[bench]
fn query_range_20e3_32(b: &mut Bencher) {
    let aabb = Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(20, 20, 20));
    let mut points = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            for z in 0..20 {
                points.push(TileData {
                    point: Vector3::new(x, y, z),
                });
            }
        }
    }
    let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 128>::with_capacity(
        250,
    )));
    let octree = Octree::<i32, TileData, 128>::new(aabb, None, container);
    octree.insert_elements(&points.clone()).ok();

    b.iter(|| {
        octree.query_range(aabb);
    });
}

#[bench]
fn query_point_20e3_32(b: &mut Bencher) {
    let aabb = Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(20, 20, 20));
    let mut points = Vec::new();
    for x in 0..20 {
        for y in 0..20 {
            for z in 0..20 {
                points.push(TileData {
                    point: Vector3::new(x, y, z),
                });
            }
        }
    }
    let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 128>::with_capacity(
        250,
    )));
    let octree = Octree::<i32, TileData, 128>::new(aabb, None, container);
    octree.insert_elements(&points.clone()).ok();

    b.iter(|| {
        points.iter().for_each(|p| {
            octree.query_point(&p.get_point());
        });
    });
}
