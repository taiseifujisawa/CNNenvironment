def downsampling(array, x, y, rate, x_max, y_max, thickness=1.0):
  assert rate > 0, "Down sampling rate should be more than 0"
  def leftlimit():
    if x - rate < 0:
      return 1
    else:
      return 0
  def rightlimit():
    if x + rate >= x_max:
      return 2
    else:
      return 0
  def uplimit():
    if y - rate < 0:
      return 4
    else:
      return 0
  def downlimit():
    if y + rate >= y_max:
      return 8
    else:
      return 0
  def limitcase():
    return leftlimit() + rightlimit() + uplimit() + downlimit()

  if limitcase() == 0:
    array[x-rate:x, y-rate:y] = thickness   # leftabove
    array[x:x+rate, y-rate:y] = thickness   # rightabove
    array[x-rate:x, y:y+rate] = thickness   # leftbelow
    array[x:x+rate, y:y+rate] = thickness   # rightbelow
  if limitcase() == 1:
    array[:x, y-rate:y] = thickness
    array[x:x+rate, y-rate:y] = thickness
    array[:x, y:y+rate] = thickness
    array[x:x+rate, y:y+rate] = thickness
  if limitcase() == 2:
    array[x-rate:x, y-rate:y] = thickness
    array[x:, y-rate:y] = thickness
    array[x-rate:x, y:y+rate] = thickness
    array[x:, y:y+rate] = thickness
  if limitcase() == 3:
    array[:x, y-rate:y] = thickness
    array[x:, y-rate:y] = thickness
    array[:x, y:y+rate] = thickness
    array[x:, y:y+rate] = thickness
  if limitcase() == 4:
    array[x-rate:x, :y] = thickness
    array[x:x+rate, :y] = thickness
    array[x-rate:x, y:y+rate] = thickness
    array[x:x+rate, y:y+rate] = thickness
  if limitcase() == 5:
    array[:x, :y] = thickness
    array[x:x+rate, :y] = thickness
    array[:x, y:y+rate] = thickness
    array[x:x+rate, y:y+rate] = thickness
  if limitcase() == 6:
    array[x-rate:x, :y] = thickness
    array[x:, :y] = thickness
    array[x-rate:x, y:y+rate] = thickness
    array[x:, y:y+rate] = thickness
  if limitcase() == 7:
    array[:x, :y] = thickness
    array[x:, :y] = thickness
    array[:x, y:y+rate] = thickness
    array[x:, y:y+rate] = thickness
  if limitcase() == 8:
    array[x-rate:x, y-rate:y] = thickness
    array[x:x+rate, y-rate:y] = thickness
    array[x-rate:x, y:] = thickness
    array[x:x+rate, y:] = thickness
  if limitcase() == 9:
    array[:x, y-rate:y] = thickness
    array[x:x+rate, y-rate:y] = thickness
    array[:x, y:] = thickness
    array[x:x+rate, y:] = thickness
  if limitcase() == 10:
    array[x-rate:x, y-rate:y] = thickness
    array[x:, y-rate:y] = thickness
    array[x-rate:x, y:] = thickness
    array[x:, y:] = thickness
  if limitcase() == 11:
    array[:x, y-rate:y] = thickness
    array[x:, y-rate:y] = thickness
    array[:x, y:] = thickness
    array[x:, y:] = thickness
  if limitcase() == 12:
    array[x-rate:x, :y] = thickness
    array[x:x+rate, :y] = thickness
    array[x-rate:x, y:] = thickness
    array[x:x+rate, y:] = thickness
  if limitcase() == 13:
    array[:x, :y] = thickness
    array[x:x+rate, :y] = thickness
    array[:x, y:] = thickness
    array[x:x+rate, y:] = thickness
  if limitcase() == 14:
    array[x-rate:x, :y] = thickness
    array[x:, :y] = thickness
    array[x-rate:x, y:] = thickness
    array[x:, y:] = thickness
  if limitcase() == 15:
    array[:x, :y] = thickness
    array[x:, :y] = thickness
    array[:x, y:] = thickness
    array[x:, y:] = thickness

  return array
