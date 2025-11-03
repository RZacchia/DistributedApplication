using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BookRent.Identity.Models;

[Table("UserCredentials")]
public class UserCredentials
{
    [Key]
    public Guid UserId { get; set; }
    [Required, MaxLength(50)]
    public required string Email { get; set; }
    [MaxLength(100)]
    public required string Password { get; set; }

    public string RefreshToken { get; set; } = string.Empty;
}