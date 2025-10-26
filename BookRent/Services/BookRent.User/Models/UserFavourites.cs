using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace BookRent.User.Models;

[Table("UserFavourites")]
[PrimaryKey(nameof(UserId), nameof(BookId))]
public class UserFavourites
{
    public Guid UserId { get; set; }
    public Guid BookId { get; set; }
}