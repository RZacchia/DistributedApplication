using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Identity.Models;
[Table("UserRole")]
public class UserRole
{
    [Key]
    public Guid UserId { get; set; }
    public Role Role { get; set; }
}