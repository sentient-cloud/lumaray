use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

/// A hashmap that stores key and values with a unique index for faster access.
///
/// Does not allow for removal of elements.
pub struct HashindMap<K, V> {
    keys: HashMap<u64, usize>, // key hash -> index
    real_keys: Vec<K>,         // index -> key
    values: Vec<(u64, V)>,     // index -> (key hash, value)
    _marker: std::marker::PhantomData<K>,
}

impl<K, V> HashindMap<K, V>
where
    K: Hash,
{
    pub fn new() -> Self {
        Self {
            keys: HashMap::with_capacity(32),
            values: Vec::with_capacity(32),
            real_keys: Vec::with_capacity(32),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> usize {
        let index = self.values.len();

        let mut hash = DefaultHasher::new();
        key.hash(&mut hash);
        let hash = hash.finish();

        self.keys.insert(hash, index);
        self.values.push((hash, value));
        self.real_keys.push(key);

        index
    }

    pub fn index_of(&self, key: K) -> Option<usize> {
        let mut hash = DefaultHasher::new();
        key.hash(&mut hash);
        let hash = hash.finish();

        self.keys.get(&hash).copied()
    }

    pub fn get(&self, index: usize) -> Option<&V> {
        self.values.get(index).map(|(_, value)| value)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut V> {
        self.values.get_mut(index).map(|(_, value)| value)
    }

    pub fn get_by_key(&self, key: K) -> Option<&V> {
        let mut hash = DefaultHasher::new();
        key.hash(&mut hash);
        let hash = hash.finish();

        self.keys
            .get(&hash)
            .and_then(|&index| self.values.get(index))
            .map(|(_, value)| value)
    }

    pub fn get_mut_by_key(&mut self, key: K) -> Option<&mut V> {
        let mut hash = DefaultHasher::new();
        key.hash(&mut hash);
        let hash = hash.finish();
        self.values
            .get_mut(*self.keys.get(&hash)?)
            .map(|(_, value)| value)
    }

    pub fn lookup_key(&self, index: usize) -> Option<&K> {
        self.real_keys.get(index)
    }

    pub fn lookup_index(&self, key: K) -> Option<usize> {
        let mut hash = DefaultHasher::new();
        key.hash(&mut hash);
        let hash = hash.finish();

        self.keys.get(&hash).copied()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns an iterator over the key-value pairs in the map, in the order they were inserted.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.real_keys
            .iter()
            .zip(self.values.iter().map(|(_, value)| value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hashindmap() {
        let mut map = HashindMap::new();

        let index = map.insert("hello", 42);

        println!("{:?} {:?}", map.get(index), Some(&42));
        println!("{:?} {:?}", map.get_by_key("hello"), Some(&42));
        println!("{:?} {:?}", map.lookup_key(index), Some(&"hello"));
        println!("{:?} {:?}", map.lookup_index("hello"), Some(index));
    }
}
