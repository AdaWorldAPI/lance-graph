# North stars — map and teleport

The planner does not build the route. It draws a map. The terminal teleports to the measurement the map names. Each star is a landing the teleport is allowed to make. A landing that is not on this list is a refusal.

Scaffold: `../scaffold/best_query.rs`. Law: `../../../../05_query_languages/associative_collapse.md`. Capstone: `../../CLAUDE_LANE_FOLD_CAPSTONE.md`.

## The map

A map is a borrowed aperture, a shift or one alignment, and a set of requests. It is not a SQL string and not a pair list. `Query` in the scaffold is the map. `collapse` is the map being redrawn so a thousand identical requests occupy one point.

## The teleport

A teleport is one terminal per homomorphism class. It writes the count, the sum, the min, the max, or the exists-bit. It does not write the route. `BestQuery.terminals` is the set of teleports. `remaining` is how many landings fired. `collapsed` is how many requests never left the map.

## Stars

| star | landing | not a landing |
|---|---|---|
| Rail | tick `i` is tick `i+d` | a zipper |
| Aperture | one 8 KB mask, borrowed | a million masks |
| Homomorphism | one terminal per `(aperture, op, lane)` | a thousand walks |
| Average | one sum and one count, divided once | averaged averages |
| Complement | left, anti, mark are the same mask | three joins |
| Span | a range on the ordered rail is a `u16` interval | a compare per row |
| Zone | a tile of 256 zero words is a continue | a word skip claimed as a tile skip |
| Proof | the checksum is a fold of the same aperture | a row vector that is then hashed |
| Refusal | a zip, a string, a nested loop, a cyclic hop, rows asked for | a silent hash table |

## How a session uses them

Pick the star the edit lands on. Name it in the commit. An edit that lands on none of them does not belong in the fold. The cognitive layers are not a star.
