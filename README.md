# Julia

The Mandelbrot set is the set of complex numbers $c$, for which this iterative function converges:

$$z_0 = 0, \qquad z_{n+1} = z_n^2 + c$$

For each $c$, there is a corresponding Julia set of the values the $z_0$ that will cause this function to converge.

It is also possible to iterate this function with the quaternion numbers, because the only multiplication performed is the squaring of the previous value.

This program shows the Mandelbrot set alonside the Julia set generated by these $c$ values, with values taken from quaternion space.

##Coordinates

See an in-depth explanation of the Coordinate system [here](coordinates.pdf).

###Changing the value of c

Keyboard keys <kbd>A</kbd> to increase and <kbd>Z</kbd> to decrease the real value of c.  m[0]
![increment_m0](https://github.com/themaninorange/Julia/blob/master/gifs/increment_m0.gif "Move c in the positive real direction.")

Keyboard keys <kbd>S</kbd> to increase and <kbd>X</kbd> to decrease the i-component of c.  m[1]
![decrement_m1](https://github.com/themaninorange/Julia/blob/master/gifs/decrement_m1.gif "Move c in the negative i direction.")

Keyboard keys <kbd>D</kbd> to increase and <kbd>C</kbd> to decrease the j-component of c.  m[2]
![increment_m2](https://github.com/themaninorange/Julia/blob/master/gifs/increment_m2.gif "Move c in the positive j direction.")

Keyboard keys <kbd>F</kbd> to increase and <kbd>V</kbd> to decrease the k-component of c.  m[3]
![increment_m3](https://github.com/themaninorange/Julia/blob/master/gifs/increment_m3.gif "Move c in the positive k direction.")

Keyboard keys <kbd>G</kbd> to increase and <kbd>B</kbd> to decrease the amount that the previous eight controls change their values.  Large values of this hidden variable correspond to large movements when adjusting c, and small values correspond to small movements when adjusting c.

Click and drag the Mandelbrot side view to move c in the real and i directions.

###Changing the center of the Julia-side view.

Keyboard keys <kbd>1</kbd> to increase and <kbd>Q</kbd> to decrease the real component of our viewing center.  c[0]
Keyboard keys <kbd>2</kbd> to increase and <kbd>W</kbd> to decrease the i-component of our viewing center.  c[1]
![increment_c01](https://github.com/themaninorange/Julia/blob/master/gifs/increment_c01.gif "Move center of view in the positive real and i directions.")
Note: These two motions are combined to show the diagonal motion.

Keyboard keys <kbd>3</kbd> to increase and <kbd>E</kbd> to decrease the j-component of our viewing center.  c[2]
![increment_c2](https://github.com/themaninorange/Julia/blob/master/gifs/increment_c2.gif "Move center of view in the positive k directions.")
Note: These two motions are combined to show the diagonal motion.

Keyboard keys <kbd>4</kbd> to increase and <kbd>R</kbd> to decrease the k-component of our viewing center.  c[3]


###Changing the horizontal direction of the Julia-side view.

From the center of the view to the center of the right edge of the view, establish a vector.  This vector has four dimensions.

Keyboard keys <kbd>5</kbd> to increase and <kbd>T</kbd> to decrease the real component of the horizontal vector.  v[0]
Keyboard keys <kbd>6</kbd> to increase and <kbd>Y</kbd> to decrease the i-component of the horizontal vector.  v[1]
![move_v01](https://github.com/themaninorange/Julia/blob/master/gifs/move_v01.gif "Adjust the horizontal vector from the positive real direction to the negative real and negative i direction.  It will appear to shrink and spin in the same way that this vector decreases in magnitude and spins.  Note: These two motions are combined to show the diagonal motion of this vector")
Keyboard keys <kbd>7</kbd> to increase and <kbd>U</kbd> to decrease the j-component of the horizontal vector.  v[2]
![move_v13](https://github.com/themaninorange/Julia/blob/master/gifs/move_v13.gif "Adjust the horizontal vector from the positive real direction to the positive j direction.  As the view changes, we see different slices through this 4-dimensional space.  Note: These two motions are combined to show the diagonal motion.")

Keyboard keys <kbd>8</kbd> to increase and <kbd>I</kbd> to decrease the k-component of the horizontal vector.  v[3]

###Changing the vertical direction of the Julia-side view.

From the center of the view to the center of the top of the view, establish a vector.  This vector has four dimensions.  However, because the horizontal vector was established first, this vector must be from a smaller subset of this space.  For more details, [see here](coordinates.pdf).  The end result is that we can control this vector with only two dimensions, analogous to latitude and longitude.

Keyboard keys <kbd>9</kbd> to increase and <kbd>O</kbd> (The upper-case letter o) to decrease the latitude of this vector.  coordinates[0]
![move_o0](https://github.com/themaninorange/Julia/blob/master/gifs/move_o0.gif "Adjust the vertical vector to increase in latitude.  The resulting vector will point in a direction that depends on the value of the horizontal vector.  It will always have the same length as our horizontal vector, and it will always be orthogonal to it.")

Keyboard keys <kbd>0</kbd> to increase and <kbd>P</kbd> to decrease the longitude of this vector.  coordinates[1]
![move_o1](https://github.com/themaninorange/Julia/blob/master/gifs/move_o1.gif "Adjust the vertical vector to increase in longitude.  The resulting vector will point in a direction that depends on the value of the horizontal vector.  It will always have the same length as our horizontal vector, and it will always be orthogonal to it.")

Keyboard keys <kbd>-</kbd> to increase and <kbd>[</kbd> to decrease the amount that the previous twenty controls change their values.  Large values of this hidden variable correspond to large movements when adjusting the view in any way, and small values correspond to small view movements.

###Quality of life view adjustments

Keyboard keys <kbd>=</kbd> to zoom in and <kbd>B</kbd> to zoom out.

Click and drag the Julia side view to move the center in the directions currently aligned with the view.






