using BookRent.Identity.DTOs.Enums;

namespace BookRent.Identity.DTOs;

public record LoginRequest(string UserName, string Password);
public record RegisterOnStoreRequest(Guid Id, string Email, string Password, Role Role);