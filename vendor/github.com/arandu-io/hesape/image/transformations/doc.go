// Package transformations holds the twelve things that can be asked of an
// image, as twelve types.
//
// Each is a record and nothing else: plain exported fields, with no
// behaviour of their own. What they mean is decided by the driver that
// executes them, and the driver in the image package is the one that decides
// it here.
//
// They live in their own package for one reason: the image package builds
// them, so if [Transformation] lived there, this package could not name it
// and the import would be a cycle.
//
// A caller rarely names these types directly. The image package's Image type
// has a method for each -- Cover, Blur, Orient -- and they are the way in;
// what these are for is Image.Transform, and the custom transformation that a
// driver was taught to handle with ImageManager.TransformUsing.
package transformations
