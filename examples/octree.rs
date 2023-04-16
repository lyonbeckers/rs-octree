use std::sync::Arc;

use aabb::Aabb;
use nalgebra::Vector3;
use octree::{Octree, OctreeVec, PointData};
use parking_lot::RwLock;
use tracing::Subscriber;
use tracing_subscriber::EnvFilter;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TileData {
    point: Vector3<i32>,
}

impl PointData<i32> for TileData {
    fn get_point(&self) -> Vector3<i32> {
        self.point
    }
}

fn setup_subscriber() -> impl Subscriber {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .finish()
}

pub fn main() {
    tracing::subscriber::set_global_default(setup_subscriber()).ok();

    let container = Arc::new(RwLock::new(OctreeVec::<i32, TileData, 8>::new()));
    let octree = Octree::<i32, TileData, 8>::new(
        Aabb::from_extents(Vector3::new(0, 0, 0), Vector3::new(10, 10, 10)),
        None,
        container,
    );

    let mut tiles = vec![];
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let tile = TileData {
                    point: Vector3::new(x, y, z),
                };
                octree.write().insert(tile).ok();
                tiles.push(tile);
            }
        }
    }
    for x in 4..10 {
        for y in 4..10 {
            for z in 4..10 {
                let tile = TileData {
                    point: Vector3::new(x, y, z),
                };
                octree.write().insert(tile).ok();
                tiles.push(tile);
            }
        }
    }
}
