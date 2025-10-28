using BookRent.Identity.DTO;
using BookRent.Identity.DTO.Enums;
using BookRent.Identity.Infrastructure.Interfaces;
using BookRent.Identity.Models;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using LoginRequest = Microsoft.AspNetCore.Identity.Data.LoginRequest;

namespace BookRent.Identity.Api;

internal static class IdentityEndpoints
{
    internal static async Task<Results<Ok<string>, BadRequest>> Login([FromBody] LoginRequest request, IIdentityRepository repo)
    {
        var id = await repo.GetUserIdAsync(request.Email, request.Password);
        if (id == null) return TypedResults.BadRequest();
        return TypedResults.Ok(id.ToString());
    }
    
    internal static async Task<IResult> Register([FromBody] RegisterOnStoreRequest request, IIdentityRepository repo)
    {
        var userRole = new UserRole
        {
            UserId = request.Id,
            Role = request.Role
        };
        var userCred = new UserCredentials
        {
            UserId = request.Id,
            Email = request.Email,
            Password = request.Password
        };
        var success = await repo.RegisterUserAsync(userCred,  userRole);
        if (!success) return Results.BadRequest();
        return Results.Ok();
    }
    
    internal static async Task<Results<Ok<Role?>, BadRequest>> GetUserRole(Guid id, IIdentityRepository repo)
    {
        var result = await repo.GetUserRoleAsync(id);
        if (result == null) return TypedResults.BadRequest();
        return TypedResults.Ok(result);
    }
}