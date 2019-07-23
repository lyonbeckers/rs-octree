pub mod common {

    pub mod units {

        pub struct Point {
            pub x: f32,
            pub y: f32,
            pub z: f32
        }

        impl std::default::Default for Point {
            fn default() -> Self {
                Point {
                    x: std::f32::MAX,
                    y: std::f32::MAX,
                    z: std::f32::MAX
                }
            }
        }

        impl std::fmt::Display for Point {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "({}, {}, {})", self.x, self.y, self.z)
            }
        }

        impl PartialEq for Point {
            fn eq(&self, rhs : &Point) -> bool {
                self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
            }
        }

        impl std::ops::Add for Point {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                Self {
                    x: self.x + other.x,
                    y: self.y + other.y,
                    z: self.z + other.z
                }
            }
        }

        impl std::ops::Sub for Point {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                Self {
                    x: self.x - other.x,
                    y: self.y - other.y,
                    z: self.z - other.z
                }
            }
        }

        impl std::ops::Div<f32> for Point {
            type Output = Self;

            fn div(self, rhs: f32) -> Self {
                Self {
                    x: self.x / rhs,
                    y: self.y / rhs,
                    z: self.z / rhs
                }
            }
        }

        impl Copy for Point {

        }

        impl Clone for Point {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl Point {

            pub fn new(x :f32, y :f32, z :f32) -> Self {
                Point {
                    x, y, z
                }
            }

        }

    }

    pub mod data_structures {

        use super::units::Point;

        pub trait PointData : Copy {
            fn get_point(&self) -> Point;
        }

        pub struct Voxel {
            point: Point
        }

        impl PointData for Voxel {
            fn get_point(&self) -> Point {
                self.point
            }
        }

        impl Default for Voxel {
            fn default() -> Voxel {
                Voxel {
                    point: Point::default()
                }
            }
        }

        impl Copy for Voxel {

        }

        impl Clone for Voxel {
            fn clone(&self) -> Voxel {
                *self
            }
        }

        impl Voxel {
            pub fn new(point: Point) -> Self {
                Voxel {
                    point: point
                }
            }

            pub fn get_point(&self) -> Point {
                self.point
            }
        }
    }

    pub mod geometry {

        use super::units::Point;

        pub struct AABB {
            pub center: Point,
            pub half_dimension: f32,
            min: Point,
            max: Point
        }

        impl Copy for AABB {

        }

        impl Clone for AABB {
            fn clone(&self) -> AABB {
                *self
            }
        }

        impl AABB {
            pub fn new(center: Point, half_dimension: f32) -> AABB {
                AABB {
                    center, 
                    half_dimension,
                    max: Point::new(center.x + half_dimension, 
                        center.y + half_dimension,
                        center.z + half_dimension),
                    min: Point::new(center.x - half_dimension, 
                        center.y - half_dimension,
                        center.z - half_dimension)
                }
            }

            pub fn get_max(&self) -> Point {
                self.max
            }

            pub fn get_min(&self) -> Point {
                self.min
            }

            pub fn contains_point(&self, point: Point) -> bool {
                point.x >= self.min.x && point.x <= self.max.x
                && point.y >= self.min.y && point.y <= self.max.y
                && point.z >= self.min.z && point.z <= self.max.z
            }

            pub fn intersects_bounds(&self, aabb: AABB) -> bool {
                self.min.x <= aabb.max.x && self.max.x >= aabb.min.x
                && self.min.y <= aabb.max.y && self.max.y >= aabb.min.y
                && self.min.z <= aabb.max.z && self.max.z >= aabb.min.z
            }
        }

    }

}

mod collections {

    pub mod point_data {
    
        use crate::common::geometry::AABB;
        use crate::common::data_structures::PointData;
        use crate::common::units::Point;

        enum Paternity {
            ProudParent,
            ChildFree
        }

        pub struct Octree <T: PointData>{
            aabb: AABB,
            num_elements: u32,
            elements: [Option<T>; 8],
            children: Vec<Option<Octree<T>>>,
            paternity: Paternity
        }

        impl<T: PointData> Octree<T> {
            const MAX_SIZE: u32 = 8;

            pub fn new(aabb: AABB) -> Octree<T> {
                Octree {
                    aabb,
                    num_elements: 0,
                    elements: [Option::<T>::None; 8], //TODO: use option here so we don't have to use Voxel:default
                    children: vec![None, None, None, None, 
                                    None, None, None, None], //Going ahead and allocating for the vector
                    paternity: Paternity::ChildFree
                }
            }

            fn subdivide(&mut self) {
                
                println!("Subdividing {} ", self.aabb.center);

                let downbackleft = AABB::new(
                    self.aabb.center + (self.aabb.get_min() - self.aabb.center)/2.0
                    , self.aabb.half_dimension/2.0);

                let downbackright = AABB::new(
                    self.aabb.center + (Point::new(self.aabb.get_max().x, self.aabb.get_min().y, self.aabb.get_min().z) - self.aabb.center)/2.0
                    , self.aabb.half_dimension/2.0);

                let downforwardleft = AABB::new(
                    self.aabb.center + (Point::new(self.aabb.get_min().x, self.aabb.get_min().y, self.aabb.get_max().z) - self.aabb.center)/2.0
                    , self.aabb.half_dimension/2.0);

                let downforwardright = AABB::new(
                    self.aabb.center + (Point::new(self.aabb.get_max().x, self.aabb.get_min().y, self.aabb.get_max().z) - self.aabb.center)/2.0
                    , self.aabb.half_dimension/2.0);

                let upbackleft = AABB::new(
                    self.aabb.center + (Point::new(self.aabb.get_min().x, self.aabb.get_max().y, self.aabb.get_min().z) - self.aabb.center)/2.0
                    , self.aabb.half_dimension/2.0);

                let upbackright = AABB::new(
                    self.aabb.center + (Point::new(self.aabb.get_max().x, self.aabb.get_max().y, self.aabb.get_min().z) - self.aabb.center)/2.0
                    , self.aabb.half_dimension/2.0);

                let upforwardleft = AABB::new(
                    self.aabb.center + (Point::new(self.aabb.get_min().x, self.aabb.get_max().y, self.aabb.get_max().z) - self.aabb.center)/2.0
                    , self.aabb.half_dimension/2.0);

                let upforwardright = AABB::new(
                    self.aabb.center + (self.aabb.get_max() - self.aabb.center)/2.0
                    , self.aabb.half_dimension/2.0);

                self.children[0] = Some(Octree::new(downbackleft));
                self.children[1] = Some(Octree::new(downbackright));
                self.children[2] = Some(Octree::new(downforwardleft));
                self.children[3] = Some(Octree::new(downforwardright));
                self.children[4] = Some(Octree::new(upbackleft));
                self.children[5] = Some(Octree::new(upbackright));
                self.children[6] = Some(Octree::new(upforwardleft));
                self.children[7] = Some(Octree::new(upforwardright));
                
                self.paternity = Paternity::ProudParent;

            }

            pub fn insert(&mut self, element: T) -> bool{  

                if !self.aabb.contains_point(element.get_point()) {
                    return false
                }

                
                match &self.paternity { //do first match because you still need to insert into children after subdividing, not either/or

                    Paternity::ChildFree if self.num_elements < Octree::<T>::MAX_SIZE => {
                        self.elements[self.num_elements as usize] = Some(element);
                        println!("Inserted {} as element number {} in {}, {}", self.elements[self.num_elements as usize].unwrap().get_point(), self.num_elements, self.aabb.center, self.aabb.half_dimension);

                        self.num_elements = self.num_elements + 1;
                    }

                    Paternity::ChildFree => { self.subdivide(); }
                    
                    _ => {}

                }

                return match &self.paternity {

                    Paternity::ProudParent => {
                        for i in 0..self.children.len() {
                            
                            if self.children[i].as_mut().unwrap().insert(element) == true {
                                return true
                            } 
                                 
                        }
                        false
                    }

                    _ => false
                }
            }

            pub fn query_range(&mut self, range: AABB) -> Vec<T> {

                let mut elements_in_range = vec![];
                
                if !self.aabb.intersects_bounds(range) {
                    return elements_in_range
                }

                if self.num_elements == 0 {
                    return elements_in_range
                }

                for &element in self.elements.iter() {

                    match element {
                        None => continue,
                        Some(_) => {
                            let el = element.unwrap();
                            if self.aabb.contains_point(el.get_point()) {
                                elements_in_range.push(el);
                            }
                        } 
                    }
                }

                if let Paternity::ChildFree = self.paternity {
                    return elements_in_range
                }

                for child_option in &mut self.children {
                    if let Some(_) = child_option {
                        elements_in_range.append(&mut child_option.as_mut().unwrap().query_range(range));
                    }
                }

                elements_in_range
            }
        }

    }
}
