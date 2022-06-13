def downsampling(array, x, y, rate, x_max, y_max):
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
    array[x-rate:x, y-rate:y] = 0   # leftabove
    array[x:x+rate, y-rate:y] = 0   # rightabove
    array[x-rate:x, y:y+rate] = 0   # leftbelow
    array[x:x+rate, y:y+rate] = 0   # rightbelow
  if limitcase() == 1:
    array[:x, y-rate:y] = 0
    array[x:x+rate, y-rate:y] = 0
    array[:x, y:y+rate] = 0
    array[x:x+rate, y:y+rate] = 0
  if limitcase() == 2:
    array[x-rate:x, y-rate:y] = 0
    array[x:, y-rate:y] = 0
    array[x-rate:x, y:y+rate] = 0
    array[x:, y:y+rate] = 0
  if limitcase() == 3:
    array[:x, y-rate:y] = 0
    array[x:, y-rate:y] = 0
    array[:x, y:y+rate] = 0
    array[x:, y:y+rate] = 0
  if limitcase() == 4:
    array[x-rate:x, :y] = 0
    array[x:x+rate, :y] = 0
    array[x-rate:x, y:y+rate] = 0
    array[x:x+rate, y:y+rate] = 0
  if limitcase() == 5:
    array[:x, :y] = 0
    array[x:x+rate, :y] = 0
    array[:x, y:y+rate] = 0
    array[x:x+rate, y:y+rate] = 0
  if limitcase() == 6:
    array[x-rate:x, :y] = 0
    array[x:, :y] = 0
    array[x-rate:x, y:y+rate] = 0
    array[x:, y:y+rate] = 0
  if limitcase() == 7:
    array[:x, :y] = 0
    array[x:, :y] = 0
    array[:x, y:y+rate] = 0
    array[x:, y:y+rate] = 0
  if limitcase() == 8:
    array[x-rate:x, y-rate:y] = 0
    array[x:x+rate, y-rate:y] = 0
    array[x-rate:x, y:] = 0
    array[x:x+rate, y:] = 0
  if limitcase() == 9:
    array[:x, y-rate:y] = 0
    array[x:x+rate, y-rate:y] = 0
    array[:x, y:] = 0
    array[x:x+rate, y:] = 0
  if limitcase() == 10:
    array[x-rate:x, y-rate:y] = 0
    array[x:, y-rate:y] = 0
    array[x-rate:x, y:] = 0
    array[x:, y:] = 0
  if limitcase() == 11:
    array[:x, y-rate:y] = 0
    array[x:, y-rate:y] = 0
    array[:x, y:] = 0
    array[x:, y:] = 0
  if limitcase() == 12:
    array[x-rate:x, :y] = 0
    array[x:x+rate, :y] = 0
    array[x-rate:x, y:] = 0
    array[x:x+rate, y:] = 0
  if limitcase() == 13:
    array[:x, :y] = 0
    array[x:x+rate, :y] = 0
    array[:x, y:] = 0
    array[x:x+rate, y:] = 0
  if limitcase() == 14:
    array[x-rate:x, :y] = 0
    array[x:, :y] = 0
    array[x-rate:x, y:] = 0
    array[x:, y:] = 0
  if limitcase() == 15:
    array[:x, :y] = 0
    array[x:, :y] = 0
    array[:x, y:] = 0
    array[x:, y:] = 0

  return array
